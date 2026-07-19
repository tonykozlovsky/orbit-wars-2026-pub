import logging
import os
import random
import time
import timeit
import traceback
import queue
from collections import deque
from pathlib import Path
from types import SimpleNamespace
import setproctitle
import torch


from .learn import (
   learn,
   get_reward_ema_state,
   load_reward_ema_state,
)
from .learn_value import learn_value
from .losses_func_selfplay import set_popart_shared_dict, get_popart_state, load_popart_state
from ...models.models import create_impala_model
from ...gym.wall_tree_profiler import WallTreeProfiler, profiler_span
from .common import (
   CheckpointConfig,
   StopRequested,
   RollingImmediateWaitLogger,
   _LocalDirCheckpointReader,
   _checkpoint_reader_from_cfg,
   get_checkpoint_file,
   entropy_head_values_to_dict,
   entropy_ipc_tuple_for_resume,
   checkpoint_model_config_from_checkpoint_or_fallback,
   load_model_from_checkpoint,
   load_model_from_resume_sources,
   load_state_dict_as_much_as_possible,
   load_state_dict_strict,
   state_dict_fully_matches_model,
   compile_impala_model_for_rl,
   configure_process_cpu_thread_limits,
   build_optimizer_from_config,
   queue_get_or_stop,
   timed_queue_get_or_stop,
   queue_put_or_stop,
   raise_if_stop_requested,
   set_stop_event_with_reason,
)


# Matches filenames expected by common.get_checkpoint_file(..., 'latest').
CHECKPOINT_FILE_PREFIX = "checkpoint"


_LEARNER_WALL_TREE_SUMMARY_EVERY_BATCHES = 100




def _state_dict_to_cpu(state_dict: dict) -> dict:
   cpu_state_dict = {}
   for key, value in state_dict.items():
       assert isinstance(value, torch.Tensor), (
           f"model state_dict values must be tensors, got {type(value)} at {key}"
       )
       cpu_state_dict[key] = value.detach().cpu().clone()
   return cpu_state_dict




def _plain_config(obj):
   if isinstance(obj, SimpleNamespace):
       return {k: _plain_config(v) for k, v in vars(obj).items()}
   if isinstance(obj, dict):
       return {k: _plain_config(v) for k, v in obj.items()}
   if isinstance(obj, list):
       return [_plain_config(v) for v in obj]
   if isinstance(obj, tuple):
       return tuple(_plain_config(v) for v in obj)
   return obj




def _flags_with_model_config(flags, model_config: dict):
   out = SimpleNamespace(**vars(flags))
   out.model = model_config
   return out




def _main_torch_io_root(torch_io_config) -> Path:
   if isinstance(torch_io_config, dict):
       main = torch_io_config["main_torch_io"]
   else:
       main = torch_io_config.main_torch_io
   if isinstance(main, dict):
       local = main.get("local")
   else:
       local = getattr(main, "local", None)
   assert local is not None, "main_torch_io must use local storage for checkpoint I/O"
   if isinstance(local, dict):
       dirpath = local["dirpath"]
   else:
       dirpath = local.dirpath
   return Path(dirpath)




class _LocalCheckpointWriter:
   """torch.save to ``main_torch_io`` local dir (replaces external multi-backend writers)."""


   def __init__(self, root: Path) -> None:
       self._root = Path(root)
       self._root.mkdir(parents=True, exist_ok=True)


   def write_object_with_torch(self, obj: dict, name: str) -> None:
       torch.save(obj, self._root / name)




def _checkpoint_reader_writer_pair(torch_io_config):
   root = _main_torch_io_root(torch_io_config)
   root.mkdir(parents=True, exist_ok=True)
   return _LocalDirCheckpointReader(root=root), _LocalCheckpointWriter(root)




def batch_and_learn(
   flags,
   shared_steps,
   batch_queues_,
   shared_lr_lambda,
   shared_target_entropy,
   shared_shortfall_entropy,
   shared_mean_entropy_ema,
   shared_new_controller_temperature_threshold,
   stats_queue_learner_train,
   learner_free_batch_queues_,
   learner_gpu_buffers_,
   actor_weight_buffers,
   actor_weight_update_queues,
   actor_weight_ack_queue,
   popart_shared_dict,
   popart_lock,
   checkpoint_queue,
   name: str,
   stop_event,
   latest_checkpoint_steps,
   learner_gpu_id: int,
):


   setproctitle.setproctitle(name)
   configure_process_cpu_thread_limits()


   try:
       gpu_id = int(learner_gpu_id)


       torch.cuda.set_device(gpu_id)
       torch.set_default_device(f'cuda:{gpu_id}')


       logging.info(
           f'BATCH AND LEARN (train), gpu_id: {gpu_id}'
       )


       device = f"cuda:{gpu_id}"
       enable_separate_value_model = bool(flags.enable_separate_value_model)


       learner_model = create_impala_model(flags)
       load_model_from_resume_sources(
           learner_model,
           resume_checkpoint=flags.resume_checkpoint,
           load_as_much_as_possible=flags.load_as_much_as_possible,
       )
       learner_model = learner_model.to(device)
       learner_model = compile_impala_model_for_rl(learner_model)
       value_model = None
       if enable_separate_value_model:
           value_model = create_impala_model(flags)
           load_model_from_resume_sources(
               value_model,
               resume_checkpoint=flags.resume_checkpoint,
               load_as_much_as_possible=flags.load_as_much_as_possible,
           )
           value_model = value_model.to(device)
           value_model = compile_impala_model_for_rl(value_model)
       assert isinstance(actor_weight_buffers, dict), type(actor_weight_buffers)
       assert isinstance(actor_weight_update_queues, list), type(actor_weight_update_queues)
       actor_weight_pending_worker_ids: set[int] = set()
       learner_model.train()
       if enable_separate_value_model:
           assert value_model is not None
           value_model.train()
       assert len(shared_shortfall_entropy) == len(shared_target_entropy)
       assert len(shared_new_controller_temperature_threshold) == len(shared_target_entropy)
       entropy_floor_targets = torch.tensor(
           [float(shared_shortfall_entropy[i]) for i in range(len(shared_shortfall_entropy))],
           device=device,
           dtype=torch.float32,
       )
       learner_model.set_entropy_floor_targets(entropy_floor_targets)
       if enable_separate_value_model:
           assert value_model is not None
           value_model.set_entropy_floor_targets(entropy_floor_targets)


       teacher_model = None
       teacher_models_by_num_players = None
       teacher_moving_steps = 0
       moving_teacher_scope = None
       if teacher := flags.teacher:
           teacher_moving_steps = int(teacher.moving_steps)
           if teacher_moving_steps > 0:
               assert len(teacher.checkpoints) == 1, (
                   "moving teacher requires exactly one teacher checkpoint",
                   len(teacher.checkpoints),
               )
           teacher_models_by_num_players = {}
           for teacher_checkpoint in teacher.checkpoints:
               teacher_model_config = checkpoint_model_config_from_checkpoint_or_fallback(
                   teacher_checkpoint.checkpoint,
                   flags.checkpoint_model_config_fallback_checkpoint,
                   flags.model,
               )
               model = create_impala_model(
                   _flags_with_model_config(flags, teacher_model_config)
               )
               load_model_from_checkpoint(model, teacher_checkpoint.checkpoint)
               model = model.to(device)
               model.eval()
               model = compile_impala_model_for_rl(model, dynamic=True)
               model.eval()
               num_players = teacher_checkpoint.num_players
               if num_players is not None:
                   num_players = int(num_players)
               assert num_players not in teacher_models_by_num_players, num_players
               teacher_models_by_num_players[num_players] = model
               if teacher_moving_steps > 0:
                   assert teacher_model is None
                   teacher_model = model
                   moving_teacher_scope = num_players


               logging.info(
                   "Teacher model for %s loaded from checkpoint %s using "
                   "checkpoint model_config and moved to device %s",
                   "shared 2p/4p" if num_players is None else f"{num_players}p",
                   teacher_checkpoint.checkpoint.name,
                   device,
               )


       optimizers = build_optimizer_from_config(flags.optimizer_config, learner_model)
       value_optimizers = None
       if enable_separate_value_model:
           assert value_model is not None
           value_optimizers = build_optimizer_from_config(flags.value_optimizer_config, value_model)


       def lr_lambda(epoch):
           return shared_lr_lambda.value


       # Build schedulers before optional optimizer state restore so base_lrs reflect config LR,
       # and the dynamic multiplier comes solely from shared_lr_lambda.
       schedulers = [
           torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
           for optimizer in optimizers
       ]
       value_schedulers = None
       if enable_separate_value_model:
           assert value_optimizers is not None
           value_schedulers = [
               torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
               for optimizer in value_optimizers
           ]


       # Setup shared PopArt across learner processes (optional)
       if flags.enable_popart:
           set_popart_shared_dict(popart_shared_dict, popart_lock)


       moving_teacher_resume_state = None


       # Optionally resume from checkpoint: optimizer / PopArt / EMA, etc. (model weights = actor)
       if flags.resume_checkpoint:
           rc = flags.resume_checkpoint
           reader = _checkpoint_reader_from_cfg(
               CheckpointConfig(torch_io=rc.torch_io, name=rc.name)
           )
           ckpt_name = get_checkpoint_file(reader, rc.name)
           ckpt_file = reader.read_torch(ckpt_name, map_location='cpu',  weights_only=False)


           logging.info(f'Checkpoint keys: {ckpt_file.keys()}')
           if 'model_state_dict' not in ckpt_file:
               raise KeyError("Missing model_state_dict in checkpoint")
           resume_model_state_matches = state_dict_fully_matches_model(
               learner_model,
               ckpt_file['model_state_dict'],
           )
           resume_optimizer_state_allowed = bool(resume_model_state_matches)
           if enable_separate_value_model:
               assert value_model is not None
               if 'value_model_state_dict' in ckpt_file:
                   value_model_state = ckpt_file['value_model_state_dict']
                   logging.info('Loading value_model_state_dict from checkpoint')
               else:
                   value_model_state = ckpt_file['model_state_dict']
                   logging.info('value_model_state_dict missing in checkpoint; loading value model from model_state_dict')
               assert isinstance(value_model_state, dict), (
                   'value model checkpoint state must be a dict'
               )
               value_model_state_matches = state_dict_fully_matches_model(
                   value_model,
                   value_model_state,
               )
               resume_optimizer_state_allowed = (
                   resume_optimizer_state_allowed and bool(value_model_state_matches)
               )
               if bool(flags.load_as_much_as_possible):
                   load_state_dict_as_much_as_possible(value_model, value_model_state)
               else:
                   load_state_dict_strict(value_model, value_model_state)
           te, sf, ema = entropy_ipc_tuple_for_resume(
               ckpt_file,
               flags,
               int(shared_steps.value),
           )
           assert len(te) == len(shared_target_entropy)
           assert len(ema) == len(shared_mean_entropy_ema)
           assert len(sf) == len(shared_shortfall_entropy)
           for i in range(len(shared_target_entropy)):
               shared_target_entropy[i] = float(te[i])
               shared_mean_entropy_ema[i] = float(ema[i])
               shared_shortfall_entropy[i] = float(sf[i])


           reward_ema_state = ckpt_file.get('reward_ema_state')
           if reward_ema_state is not None:
               load_reward_ema_state(reward_ema_state)
               logging.info('Loaded reward_ema_state from checkpoint')
           elif bool(flags.enable_reward_ema_norm):
               logging.info('reward_ema_state missing in checkpoint; EMA reset to defaults')


           if flags.enable_popart and ('popart_state' in ckpt_file):
               logging.info(f'Loading popart state from checkpoint {flags.resume_checkpoint.name}')
               load_popart_state(ckpt_file['popart_state'])


           moving_teacher_resume_state = ckpt_file.get('moving_teacher_state')
           if not bool(resume_optimizer_state_allowed):
               logging.info(
                   'Skipping optimizer state restore from %s because model state_dict does not fully match current model',
                   flags.resume_checkpoint.name,
               )


           if not bool(resume_optimizer_state_allowed):
               pass
           elif bool(flags.load_as_much_as_possible):
               logging.info(
                   'Skipping optimizer state restore from %s because load_as_much_as_possible=True',
                   flags.resume_checkpoint.name,
               )
           elif 'optimizer_state_dict' in ckpt_file:
               logging.info(f'Loading optimizer state dict from checkpoint {flags.resume_checkpoint.name}')
               saved_opt_state = ckpt_file['optimizer_state_dict']


               if not isinstance(saved_opt_state, list):
                   saved_opt_state = [saved_opt_state]


               assert len(saved_opt_state) == len(optimizers), (
                   'checkpoint optimizer_state_dict count must match number of optimizers: '
                   f'{len(saved_opt_state)} vs {len(optimizers)}'
               )
               for optimizer, saved_state in zip(optimizers, saved_opt_state, strict=True):
                   assert isinstance(saved_state, dict), (
                       'optimizer_state_dict entry must be a dict'
                   )
                   try:
                       optimizer.load_state_dict(saved_state)
                   except ValueError as e:
                       logging.warning(
                           'Skipping optimizer state restore from %s: %s',
                           flags.resume_checkpoint.name,
                           e,
                       )
           if enable_separate_value_model:
               assert value_optimizers is not None
               if not bool(resume_optimizer_state_allowed):
                   pass
               elif bool(flags.load_as_much_as_possible):
                   logging.info(
                       'Skipping value optimizer state restore from %s because load_as_much_as_possible=True',
                       flags.resume_checkpoint.name,
                   )
               elif 'value_optimizer_state_dict' in ckpt_file:
                   logging.info(f'Loading value optimizer state dict from checkpoint {flags.resume_checkpoint.name}')
                   saved_value_opt_state = ckpt_file['value_optimizer_state_dict']


                   if not isinstance(saved_value_opt_state, list):
                       saved_value_opt_state = [saved_value_opt_state]


                   assert len(saved_value_opt_state) == len(value_optimizers), (
                       'checkpoint value_optimizer_state_dict count must match number of value optimizers: '
                       f'{len(saved_value_opt_state)} vs {len(value_optimizers)}'
                   )
                   for optimizer, saved_state in zip(value_optimizers, saved_value_opt_state, strict=True):
                       assert isinstance(saved_state, dict), (
                           'value_optimizer_state_dict entry must be a dict'
                       )
                       optimizer.load_state_dict(saved_state)
               else:
                   logging.info('value_optimizer_state_dict missing in checkpoint; value optimizer starts fresh')


       iteration = 0
       last_checkpoint_time = timeit.default_timer()
       checkpoint_reader, checkpoint_writer = _checkpoint_reader_writer_pair(
           flags.torch_io_config
       )
       checkpoint_root = _main_torch_io_root(flags.torch_io_config)
       writer = checkpoint_writer
       moving_teacher_checkpoints = deque()
       moving_teacher_loaded_steps = None
       moving_teacher_loaded_checkpoint_ref = None
       moving_teacher_kl_cost_multiplier = 1.0
       moving_teacher_model_initialized = False
       training_done = False


       get_batch_wait_logger = RollingImmediateWaitLogger(
           metric_name="batch_and_learn.get_batch",
           log_interval_sec=60.0,
       )


       def _checkpoint_steps() -> int:
           return int(shared_steps.value)


       def _checkpoint_file_name_for_steps(steps: int) -> str:
           return f'{CHECKPOINT_FILE_PREFIX}_{int(steps)}.pt'


       def _checkpoint_ref(checkpoint_file: str, steps: int) -> dict:
           checkpoint_path = (checkpoint_root / checkpoint_file).resolve()
           return {
               "steps": int(steps),
               "checkpoint_file": checkpoint_file,
               "checkpoint_path": str(checkpoint_path),
           }


       def _checkpoint_ref_from_entry(entry: dict) -> dict:
           assert "steps" in entry
           assert "checkpoint_file" in entry
           assert "checkpoint_path" in entry
           return {
               "steps": int(entry["steps"]),
               "checkpoint_file": str(entry["checkpoint_file"]),
               "checkpoint_path": str(entry["checkpoint_path"]),
           }


       def _source_checkpoint_ref_from_config(cfg: CheckpointConfig | SimpleNamespace) -> dict:
           reader = _checkpoint_reader_from_cfg(cfg)
           checkpoint_file = get_checkpoint_file(reader, cfg.name)
           checkpoint_path = (reader.root / checkpoint_file).resolve()
           return {
               "checkpoint_file": str(checkpoint_file),
               "checkpoint_path": str(checkpoint_path),
           }


       current_teacher_source_checkpoint_ref = None
       if teacher_moving_steps > 0:
           assert flags.teacher is not None
           assert len(flags.teacher.checkpoints) == 1
           current_teacher_source_checkpoint_ref = _source_checkpoint_ref_from_config(
               flags.teacher.checkpoints[0].checkpoint
           )


       def _read_checkpoint_ref(ref: dict) -> dict:
           assert "checkpoint_path" in ref
           checkpoint_path = Path(str(ref["checkpoint_path"]))
           checkpoint = torch.load(
               checkpoint_path,
               map_location="cpu",
               weights_only=False,
           )
           assert isinstance(checkpoint, dict), (
               f"moving teacher checkpoint must be a dict: {checkpoint_path}"
           )
           assert "steps" in checkpoint
           assert "model_state_dict" in checkpoint
           assert "model_config" in checkpoint
           assert int(checkpoint["steps"]) == int(ref["steps"]), (
               int(checkpoint["steps"]),
               int(ref["steps"]),
               checkpoint_path,
           )
           return checkpoint


       def _checkpoint_payload_for_moving_teacher(entry: dict) -> dict:
           if "model_state_dict" not in entry or "model_config" not in entry:
               checkpoint = _read_checkpoint_ref(entry)
               entry["model_state_dict"] = checkpoint["model_state_dict"]
               entry["model_config"] = checkpoint["model_config"]
           return entry


       def _create_teacher_model_from_model_config(model_config: dict, model_state_dict: dict):
           model = create_impala_model(_flags_with_model_config(flags, model_config))
           model = model.to(device)
           model.eval()
           model = compile_impala_model_for_rl(model)
           model.load_state_dict(model_state_dict, strict=True)
           model.eval()
           return model


       def _moving_teacher_state_for_checkpoint() -> dict | None:
           if teacher_moving_steps == 0:
               return None
           state = {
               "moving_steps": int(teacher_moving_steps),
               "source_checkpoint_ref": current_teacher_source_checkpoint_ref,
               "model_initialized": bool(moving_teacher_model_initialized),
               "loaded_steps": (
                   None
                   if moving_teacher_loaded_steps is None
                   else int(moving_teacher_loaded_steps)
               ),
               "loaded_checkpoint_ref": moving_teacher_loaded_checkpoint_ref,
               "checkpoint_refs": [
                   _checkpoint_ref_from_entry(entry)
                   for entry in moving_teacher_checkpoints
               ],
           }
           if bool(moving_teacher_model_initialized):
               assert moving_teacher_loaded_checkpoint_ref is not None
               assert moving_teacher_loaded_steps is not None
           return state


       def _restore_moving_teacher_state(state: dict | None) -> None:
           nonlocal teacher_model
           nonlocal teacher_models_by_num_players
           nonlocal moving_teacher_loaded_steps
           nonlocal moving_teacher_loaded_checkpoint_ref
           nonlocal moving_teacher_model_initialized
           if teacher_moving_steps == 0 or state is None:
               return
           if int(state["moving_steps"]) != int(teacher_moving_steps):
               logging.info(
                   "Moving teacher state reset because moving_steps changed: checkpoint=%d config=%d",
                   int(state["moving_steps"]),
                   int(teacher_moving_steps),
               )
               return
           assert current_teacher_source_checkpoint_ref is not None
           assert "source_checkpoint_ref" in state
           if state["source_checkpoint_ref"] != current_teacher_source_checkpoint_ref:
               logging.info(
                   "Moving teacher state reset because teacher checkpoint changed: checkpoint=%s config=%s",
                   state["source_checkpoint_ref"],
                   current_teacher_source_checkpoint_ref,
               )
               return
           assert "checkpoint_refs" in state
           moving_teacher_checkpoints.clear()
           for ref in state["checkpoint_refs"]:
               moving_teacher_checkpoints.append(_checkpoint_ref_from_entry(ref))
           moving_teacher_loaded_steps = state["loaded_steps"]
           if moving_teacher_loaded_steps is not None:
               moving_teacher_loaded_steps = int(moving_teacher_loaded_steps)
           moving_teacher_loaded_checkpoint_ref = state["loaded_checkpoint_ref"]
           moving_teacher_model_initialized = bool(state["model_initialized"])
           if bool(moving_teacher_model_initialized):
               assert moving_teacher_loaded_checkpoint_ref is not None
               checkpoint = _read_checkpoint_ref(moving_teacher_loaded_checkpoint_ref)
               teacher_model = _create_teacher_model_from_model_config(
                   checkpoint["model_config"],
                   checkpoint["model_state_dict"],
               )
               assert isinstance(teacher_models_by_num_players, dict)
               assert tuple(teacher_models_by_num_players.keys()) == (moving_teacher_scope,)
               teacher_models_by_num_players[moving_teacher_scope] = teacher_model
               logging.info(
                   "Moving teacher restored from checkpoint steps=%d path=%s",
                   int(checkpoint["steps"]),
                   str(moving_teacher_loaded_checkpoint_ref["checkpoint_path"]),
               )


       def _build_checkpoint_dict_at_steps(steps: int) -> dict:
           model_state = _state_dict_to_cpu(learner_model.state_dict())
           if len(optimizers) == 1:
               optimizer_state_dict = optimizers[0].state_dict()
           else:
               optimizer_state_dict = [
                   optimizer.state_dict()
                   for optimizer in optimizers
               ]
           checkpoint = {
               'model_state_dict': model_state,
               'model_config': _plain_config(flags.model),
               'optimizer_state_dict': optimizer_state_dict,
               'shared_target_entropy_by_head': entropy_head_values_to_dict(
                   shared_target_entropy
               ),
               'shared_shortfall_entropy_by_head': entropy_head_values_to_dict(
                   shared_shortfall_entropy
               ),
               'shared_mean_entropy_ema_by_head': entropy_head_values_to_dict(
                   shared_mean_entropy_ema
               ),
               'new_controller_temperature_threshold_by_head': entropy_head_values_to_dict(
                   shared_new_controller_temperature_threshold
               ),
               'steps': int(steps),
               'reward_ema_state': get_reward_ema_state(),
           }
           if enable_separate_value_model:
               assert value_model is not None
               assert value_optimizers is not None
               checkpoint['value_model_state_dict'] = _state_dict_to_cpu(
                   value_model.state_dict()
               )
               if len(value_optimizers) == 1:
                   checkpoint['value_optimizer_state_dict'] = (
                       value_optimizers[0].state_dict()
                   )
               else:
                   checkpoint['value_optimizer_state_dict'] = [
                       optimizer.state_dict()
                       for optimizer in value_optimizers
                   ]
           if flags.enable_popart:
               checkpoint['popart_state'] = get_popart_state()
           moving_teacher_state = _moving_teacher_state_for_checkpoint()
           if moving_teacher_state is not None:
               checkpoint['moving_teacher_state'] = moving_teacher_state
           return checkpoint


       def _build_checkpoint_dict() -> dict:
           return _build_checkpoint_dict_at_steps(_checkpoint_steps())


       def _publish_actor_weights() -> None:
           assert len(actor_weight_pending_worker_ids) == 0, (
               "cannot publish actor weights while previous publish is pending",
               sorted(actor_weight_pending_worker_ids),
           )
           learner_state_dict = learner_model.state_dict()
           assert set(learner_state_dict.keys()) == set(actor_weight_buffers.keys()), (
               sorted(learner_state_dict.keys()),
               sorted(actor_weight_buffers.keys()),
           )
           for key, value in learner_state_dict.items():
               dst = actor_weight_buffers[key]
               assert isinstance(value, torch.Tensor), (
                   f"model state_dict values must be tensors, got {type(value)} at {key}"
               )
               assert isinstance(dst, torch.Tensor), (
                   f"actor weight buffer values must be tensors, got {type(dst)} at {key}"
               )
               assert tuple(dst.shape) == tuple(value.shape), (
                   key,
                   tuple(dst.shape),
                   tuple(value.shape),
               )
               assert dst.dtype == value.dtype, (key, dst.dtype, value.dtype)
               dst.copy_(value.detach().to("cpu"), non_blocking=False)
           for worker_id, actor_weight_update_queue in enumerate(actor_weight_update_queues):
               actor_weight_update_queue.put_nowait(1)
               actor_weight_pending_worker_ids.add(int(worker_id))


       def _drain_actor_weight_acks() -> None:
           while True:
               try:
                   worker_id = actor_weight_ack_queue.get_nowait()
               except queue.Empty:
                   break
               worker_id = int(worker_id)
               assert worker_id in actor_weight_pending_worker_ids, (
                   worker_id,
                   sorted(actor_weight_pending_worker_ids),
               )
               actor_weight_pending_worker_ids.remove(worker_id)


       def _write_checkpoint_and_enqueue(checkpoint_file: str, checkpoint: dict) -> None:
           assert writer is not None, "Checkpoint writer is not initialized"
           assert isinstance(checkpoint["model_state_dict"], dict), (
               "checkpoint model_state_dict must be a dict"
           )
           writer.write_object_with_torch(
               checkpoint,
               checkpoint_file,
           )
           if teacher_moving_steps > 0:
               checkpoint_ref = _checkpoint_ref(
                   checkpoint_file,
                   int(checkpoint["steps"]),
               )
               moving_teacher_checkpoints.append(
                   {
                       **checkpoint_ref,
                       "model_state_dict": checkpoint["model_state_dict"],
                       "model_config": checkpoint["model_config"],
                   }
               )
           queue_put_or_stop(
               checkpoint_queue,
               {
                   "checkpoint_file": checkpoint_file,
                   "steps": int(checkpoint["steps"]),
               },
               stop_event,
               timeout_sec=0.1,
           )
           latest_checkpoint_steps.value = int(checkpoint["steps"])


       def _maybe_update_moving_teacher(current_steps: int) -> None:
           nonlocal teacher_model
           nonlocal teacher_models_by_num_players
           nonlocal moving_teacher_loaded_steps
           nonlocal moving_teacher_loaded_checkpoint_ref
           nonlocal moving_teacher_kl_cost_multiplier
           nonlocal moving_teacher_model_initialized
           if teacher_moving_steps == 0:
               return
           assert teacher_model is not None, "Moving teacher requires teacher model"
           assert isinstance(teacher_models_by_num_players, dict)
           assert tuple(teacher_models_by_num_players.keys()) == (moving_teacher_scope,)
           assert teacher_moving_steps > 0
           min_teacher_steps = int(current_steps) - int(teacher_moving_steps)
           candidates = []
           for checkpoint in moving_teacher_checkpoints:
               checkpoint_steps = int(checkpoint["steps"])
               lag_steps = int(current_steps) - checkpoint_steps
               assert lag_steps >= 0, (
                   "moving teacher checkpoint cannot be from the future",
                   checkpoint_steps,
                   int(current_steps),
               )
               if lag_steps <= int(teacher_moving_steps):
                   candidates.append(checkpoint)
           assert len(candidates) > 0, (
               "moving teacher requires at least one checkpoint inside the moving_steps window; "
               "increase moving_steps or checkpoint frequency",
               int(current_steps),
               int(teacher_moving_steps),
               [int(checkpoint["steps"]) for checkpoint in moving_teacher_checkpoints],
           )
           selected = random.choice(candidates)
           selected_steps = int(selected["steps"])
           selected_lag_steps = int(current_steps) - selected_steps
           oldest_candidate_steps = min(int(checkpoint["steps"]) for checkpoint in candidates)
           tail_lag_steps = int(current_steps) - int(oldest_candidate_steps)
           assert 0 <= tail_lag_steps <= int(teacher_moving_steps), (
               tail_lag_steps,
               int(teacher_moving_steps),
               int(current_steps),
               oldest_candidate_steps,
           )
           moving_teacher_kl_cost_multiplier = (
               0.0
               if tail_lag_steps == 0
               else float(selected_lag_steps) / float(tail_lag_steps)
           )
           assert 0.0 <= moving_teacher_kl_cost_multiplier <= 1.0, (
               selected_lag_steps,
               tail_lag_steps,
               moving_teacher_kl_cost_multiplier,
           )
           while len(moving_teacher_checkpoints) > 1:
               oldest_steps = int(moving_teacher_checkpoints[0]["steps"])
               if oldest_steps >= min_teacher_steps:
                   break
               moving_teacher_checkpoints.popleft()
           if moving_teacher_loaded_steps == selected_steps:
               return
           selected = _checkpoint_payload_for_moving_teacher(selected)
           if not bool(moving_teacher_model_initialized):
               teacher_model = _create_teacher_model_from_model_config(
                   selected["model_config"],
                   selected["model_state_dict"],
               )
               moving_teacher_model_initialized = True
               teacher_models_by_num_players[moving_teacher_scope] = teacher_model
           else:
               teacher_model.load_state_dict(selected["model_state_dict"], strict=True)
               teacher_model.eval()
           moving_teacher_loaded_steps = selected_steps
           moving_teacher_loaded_checkpoint_ref = _checkpoint_ref_from_entry(selected)
           logging.info(
               "Moving teacher loaded learner checkpoint at steps=%d current_steps=%d moving_steps=%d kl_cost_multiplier=%.6f",
               selected_steps,
               int(current_steps),
               int(teacher_moving_steps),
               moving_teacher_kl_cost_multiplier,
           )


       _restore_moving_teacher_state(moving_teacher_resume_state)


       if writer is not None:
           bootstrap_steps = _checkpoint_steps()
           checkpoint = _build_checkpoint_dict_at_steps(bootstrap_steps)
           _write_checkpoint_and_enqueue(
               _checkpoint_file_name_for_steps(bootstrap_steps),
               checkpoint,
           )
       # After bootstrap checkpoint write, publish learner weights to every inference GPU actor copy.
       _publish_actor_weights()
       wall_prof = (
           WallTreeProfiler()
           if bool(flags.enable_learner_wall_tree_profiler)
           else None
       )
       learner_wall_window_ms = 0.0
       while True:
           raise_if_stop_requested(stop_event)
           _drain_actor_weight_acks()
           iteration += 1


           t_iter_wall0 = time.perf_counter()


           batch_queue = batch_queues_[0]
           learner_free_batch_queue = learner_free_batch_queues_[0]


           with profiler_span(wall_prof, "get_batch"):
               learner_buffer_idx = timed_queue_get_or_stop(
                   batch_queue,
                   stop_event,
                   timeout_sec=0.1,
                   wait_logger=get_batch_wait_logger,
                   stream_key="train",
                   extra_context="train",
               )
               batch = learner_gpu_buffers_[learner_buffer_idx]


           with profiler_span(wall_prof, "advance_shared_steps"):
               batch_delta_steps = int(flags.batch_size) * int(flags.unroll_length)
               local_step = int(shared_steps.value)
               with shared_steps.get_lock():
                   shared_steps.value += batch_delta_steps
               learner_model.train()
               if enable_separate_value_model:
                   assert value_model is not None
                   value_model.train()


           with profiler_span(wall_prof, "moving_teacher"):
               _maybe_update_moving_teacher(local_step)


           learn_teacher_models = teacher_models_by_num_players if flags.teacher else None


           value_stats_override = None
           separate_value_baseline_learn = None
           if enable_separate_value_model:
               assert value_model is not None
               assert value_optimizers is not None
               with profiler_span(wall_prof, "learn_value"):
                   (
                       _value_targets,
                       _value_bootstrap,
                       value_stats_override,
                       separate_value_baseline_learn,
                   ) = learn_value(
                       device=device,
                       flags=flags,
                       value_model=value_model,
                       batch=batch,
                       wall_profiler=wall_prof,
                       optimizers=value_optimizers,
                       lr_schedulers=value_schedulers,
                       train=True,
                   )


           with profiler_span(wall_prof, "learn"):
               learn(
                   device=device,
                   flags=flags,
                   learner_model=learner_model,
                   batch=batch,
                   wall_profiler=wall_prof,
                   optimizers=optimizers,
                   lr_schedulers=schedulers,
                   shared_target_entropy=shared_target_entropy,
                   shared_shortfall_entropy=shared_shortfall_entropy,
                   stats_queue_learner=stats_queue_learner_train,
                   stop_event=stop_event,
                   teacher_models_by_num_players=learn_teacher_models,
                   teacher_kl_cost_multiplier=moving_teacher_kl_cost_multiplier,
                   train=True,
                   local_step=local_step,
                   stats_override=value_stats_override,
                   separate_value_baseline_learn=separate_value_baseline_learn,
               )


           sync_every = int(flags.learner_actor_sync_every_iterations)
           assert sync_every >= 1
           if (
               iteration % sync_every == 0
               and len(actor_weight_pending_worker_ids) == 0
           ):
               with profiler_span(wall_prof, "sync_actor"):
                   _publish_actor_weights()


           # Periodic checkpoint including optimizer and PopArt
           if timeit.default_timer() - last_checkpoint_time > flags.checkpoint_freq * 60:
               with profiler_span(wall_prof, "periodic_checkpoint"):
                   try:
                       checkpoint = _build_checkpoint_dict()
                       _write_checkpoint_and_enqueue(
                           _checkpoint_file_name_for_steps(checkpoint["steps"]),
                           checkpoint,
                       )
                   except Exception as e:
                       logging.info(f'Error writing checkpoint: {e}')
                       logging.info(traceback.format_exc())
               last_checkpoint_time = timeit.default_timer()


           with profiler_span(wall_prof, "return_buffer"):
               queue_put_or_stop(
                   learner_free_batch_queue,
                   learner_buffer_idx,
                   stop_event,
                   timeout_sec=0.1,
               )
          
           #try:
           #    batch_queue.put_nowait(learner_buffer_idx)
           #except queue.Full:
           #    pass


           learner_wall_window_ms += (
               time.perf_counter() - t_iter_wall0
           ) * 1000.0
           if (
               wall_prof is not None
               and iteration % _LEARNER_WALL_TREE_SUMMARY_EVERY_BATCHES == 0
           ):
               span_lo = (
                   iteration - _LEARNER_WALL_TREE_SUMMARY_EVERY_BATCHES + 1
               )
               wall_prof.summary(
                   (
                       "batch_and_learn "
                       f"batches {span_lo}-{iteration} "
                       f"(every {_LEARNER_WALL_TREE_SUMMARY_EVERY_BATCHES})"
                   ),
                   wall_ms=learner_wall_window_ms,
               )
               wall_prof.clear()
               learner_wall_window_ms = 0.0
           if shared_steps.value >= flags.total_steps:
               training_done = True
               break


       if training_done:
           if writer is not None:
               final_steps = int(shared_steps.value)
               if int(latest_checkpoint_steps.value) < final_steps:
                   final_checkpoint = _build_checkpoint_dict_at_steps(final_steps)
                   _write_checkpoint_and_enqueue(
                       _checkpoint_file_name_for_steps(final_steps),
                       final_checkpoint,
                   )
           logging.info("Learner reached total steps; waiting for stop_event")
           stop_event.wait()


  
   except KeyboardInterrupt:
       pass
   except StopRequested:
       pass
   except Exception as e:
       logging.info(traceback.format_exc())
   finally:
       set_stop_event_with_reason(
           stop_event,
           process_name=name,
           reason="batch_and_learn finally",
       )
       os._exit(0)



