#!/usr/bin/env python3
"""Poll Kaggle leaderboard, mirror replays for top leaderboard submissions only.

Только первые ``_TOP_LEADERBOARD_SUBMISSIONS`` (20) участников по ``publicLeaderboard``
в порядке API; остальные сабмиты не обходятся.

Если в ответе меньше ``_MIN_LEADERBOARD_ROWS`` (10) строк лидерборда — итерация
пропускается (ждём следующий цикл).

Эпизод считается уже сохранённым только если валидны и ``replay_<id>.json``, и
``replay_<id>_metadata.json`` (JSON с полями ``competition_id``, ``captured_at_epoch_seconds``,
``leaderboard`` — полный ответ лидерборда этой итерации, включая ``publicLeaderboard``).
Иначе реплей перекачивается и метадата перезаписывается.

За один проход главного цикла из топ-20 берётся **ровно один** submission — с минимальным
временем последнего **начала** разбора (новые id в очереди как 0): один ``ListEpisodes``,
затем все нужные скачивания по эпизодам этого сабмита, потом ``sleep``; следующий submission
будет в следующем проходе (размазывание rate limit на ``ListEpisodes``).

Словарь ``last_started_by_submission`` хранится в JSON рядом с реплеями (атомарная запись),
путь по умолчанию ``<output-dir>/kaggle_mirror_last_started_by_submission.json``.

Эндпоинты ``/api/i/competitions.*Service/...`` на стороне сайта ожидают те же заголовки,
что и страница лидерборда: как минимум ``x-kaggle-build-version`` и при логине —
``x-xsrf-token`` (значение совпадает с cookie ``XSRF-TOKEN``). Без этого часто приходит
HTTP 400 с пустым телом — это не «публичный REST без контекста браузера».

Исключения внутри одной итерации ловятся (кроме ``KeyboardInterrupt``): логируется
traceback, затем та же пауза ``--interval-minutes``, что и между успешными проходами.

Зависимость: ``pip install requests``.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from urllib.parse import unquote

import requests

_URL_GET_LEADERBOARD = (
    "https://www.kaggle.com/api/i/competitions.LeaderboardService/GetLeaderboard"
)
_URL_LIST_EPISODES = (
    "https://www.kaggle.com/api/i/competitions.EpisodeService/ListEpisodes"
)
_URL_EPISODE_REPLAY_JSON = (
    "https://www.kaggle.com/competitions/episodes/{episode_id}/replay.json"
)

_DEFAULT_REFERER = "https://www.kaggle.com/competitions/orbit-wars/leaderboard"
# Из актуального захвата DevTools (меняется при деплое фронта).
_DEFAULT_BUILD_VERSION = "7177285e912b953fa90cc731bbb971a7e110c442"
_REQUEST_PAUSE_SEC = 20.0
_MIN_LEADERBOARD_ROWS = 10
_TOP_LEADERBOARD_SUBMISSIONS = 10
_SUBMISSION_ORDER_STATE_FILENAME = "kaggle_mirror_last_started_by_submission.json"


def _cookie_value(cookie_header: str, name: str) -> str | None:
    target = name.lower()
    for part in cookie_header.split(";"):
        part = part.strip()
        if "=" not in part:
            continue
        key, _, val = part.partition("=")
        if key.strip().lower() != target:
            continue
        return unquote(val.strip())
    return None


def xsrf_from_cookie(cookie_header: str) -> str | None:
    for key in ("XSRF-TOKEN", "CSRF-TOKEN"):
        v = _cookie_value(cookie_header, key)
        if v:
            return v
    return None


def api_headers(
    *,
    referer: str,
    build_version: str,
    xsrf_token: str | None,
) -> dict[str, str]:
    h: dict[str, str] = {
        "accept": "application/json",
        "accept-language": "ru-RU,ru;q=0.9,en-US;q=0.8,en;q=0.7",
        "origin": "https://www.kaggle.com",
        "referer": referer,
        "sec-fetch-dest": "empty",
        "sec-fetch-mode": "cors",
        "sec-fetch-site": "same-origin",
        "user-agent": (
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/146.0.0.0 Safari/537.36"
        ),
        "x-kaggle-build-version": build_version,
    }
    if xsrf_token:
        h["x-xsrf-token"] = xsrf_token
    return h


def post_json(
    session: requests.Session,
    url: str,
    body: dict[str, object],
    *,
    referer: str,
    build_version: str,
    xsrf_token: str | None,
    timeout: float,
) -> dict[str, object]:
    r = session.post(
        url,
        json=body,
        headers=api_headers(
            referer=referer,
            build_version=build_version,
            xsrf_token=xsrf_token,
        ),
        timeout=timeout,
    )
    time.sleep(_REQUEST_PAUSE_SEC)
    if r.status_code != 200:
        ct = r.headers.get("Content-Type", "")
        snippet = r.text[:2000] if r.text else "(empty body)"
        assert False, (
            f"HTTP {r.status_code} {url!r} content-type={ct!r} body[:2000]={snippet!r}"
        )
    obj = r.json()
    assert isinstance(obj, dict), type(obj)
    if "result" in obj:
        inner = obj["result"]
        assert isinstance(inner, dict), type(inner)
        return inner
    return obj


def fetch_leaderboard_payload(
    session: requests.Session,
    *,
    competition_id: int,
    referer: str,
    build_version: str,
    xsrf_token: str | None,
) -> dict[str, object]:
    body = {
        "competitionId": competition_id,
        "leaderboardMode": "LEADERBOARD_MODE_DEFAULT",
    }
    payload = post_json(
        session,
        _URL_GET_LEADERBOARD,
        body,
        referer=referer,
        build_version=build_version,
        xsrf_token=xsrf_token,
        timeout=120.0,
    )
    assert "publicLeaderboard" in payload, sorted(payload.keys())
    rows = payload["publicLeaderboard"]
    assert isinstance(rows, list), type(rows)
    return payload


def submission_ids_top_n_from_leaderboard_payload(
    payload: dict[str, object],
    *,
    n: int,
) -> list[int]:
    rows_raw = payload["publicLeaderboard"]
    assert isinstance(rows_raw, list), type(rows_raw)
    out: list[int] = []
    seen: set[int] = set()
    for row in rows_raw[:n]:
        assert isinstance(row, dict), type(row)
        assert "submissionId" in row, sorted(row.keys())
        sid = row["submissionId"]
        assert isinstance(sid, int), type(sid)
        if sid in seen:
            continue
        seen.add(sid)
        out.append(sid)
    return out


def fetch_episodes_for_submission(
    session: requests.Session,
    *,
    submission_id: int,
    successful_only: bool,
    referer: str,
    build_version: str,
    xsrf_token: str | None,
) -> list[dict[str, object]]:
    body = {
        "ids": [],
        "submissionId": submission_id,
        "successfulOnly": successful_only,
    }
    payload = post_json(
        session,
        _URL_LIST_EPISODES,
        body,
        referer=referer,
        build_version=build_version,
        xsrf_token=xsrf_token,
        timeout=120.0,
    )
    assert "episodes" in payload, sorted(payload.keys())
    episodes = payload["episodes"]
    assert isinstance(episodes, list), type(episodes)
    out: list[dict[str, object]] = []
    for ep in episodes:
        assert isinstance(ep, dict), type(ep)
        assert "id" in ep, sorted(ep.keys())
        out.append(ep)
    return out


def replay_path(out_dir: Path, episode_id: int) -> Path:
    return out_dir / f"replay_{episode_id}.json"


def replay_metadata_path(out_dir: Path, episode_id: int) -> Path:
    return out_dir / f"replay_{episode_id}_metadata.json"


def replay_not_found_path(out_dir: Path, episode_id: int) -> Path:
    return out_dir / f"replay_{episode_id}_replay_json_404.json"


def episode_replay_json_url(episode_id: int) -> str:
    return _URL_EPISODE_REPLAY_JSON.format(episode_id=episode_id)


def write_json_atomic(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(obj, ensure_ascii=False, indent=2) + "\n"
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def load_submission_order_state(path: Path) -> dict[int, float]:
    if not path.is_file():
        return {}
    text = path.read_text(encoding="utf-8")
    raw = json.loads(text)
    assert isinstance(raw, dict), type(raw)
    out: dict[int, float] = {}
    for key, val in raw.items():
        assert isinstance(key, str), type(key)
        sid = int(key)
        assert type(val) is int or type(val) is float, (sid, type(val))
        out[sid] = float(val)
    return out


def save_submission_order_state(path: Path, state: dict[int, float]) -> None:
    payload = {str(sid): float(ts) for sid, ts in state.items()}
    write_json_atomic(path, payload)


def replay_file_is_valid(path: Path) -> bool:
    if not path.is_file():
        return False
    text = path.read_text(encoding="utf-8")
    try:
        raw = json.loads(text)
    except json.JSONDecodeError:
        return False
    if not isinstance(raw, dict):
        return False
    steps = raw.get("steps")
    if not isinstance(steps, list) or len(steps) < 2:
        return False
    return True


def replay_metadata_is_valid(path: Path) -> bool:
    if not path.is_file():
        return False
    text = path.read_text(encoding="utf-8")
    try:
        raw = json.loads(text)
    except json.JSONDecodeError:
        return False
    if not isinstance(raw, dict):
        return False
    cid = raw.get("competition_id")
    if type(cid) is not int:
        return False
    ts = raw.get("captured_at_epoch_seconds")
    if type(ts) is not float and type(ts) is not int:
        return False
    lb = raw.get("leaderboard")
    if not isinstance(lb, dict):
        return False
    pub = lb.get("publicLeaderboard")
    if not isinstance(pub, list):
        return False
    return True


def episode_bundle_is_complete(out_dir: Path, episode_id: int) -> bool:
    return replay_file_is_valid(replay_path(out_dir, episode_id)) and replay_metadata_is_valid(
        replay_metadata_path(out_dir, episode_id)
    )


def replay_not_found_marker_is_valid(path: Path, episode_id: int) -> bool:
    if not path.is_file():
        return False
    text = path.read_text(encoding="utf-8")
    raw = json.loads(text)
    assert isinstance(raw, dict), type(raw)
    assert raw["episode_id"] == episode_id, raw
    assert raw["status_code"] == 404, raw
    ts = raw["captured_at_epoch_seconds"]
    assert type(ts) is int or type(ts) is float, type(ts)
    return True


def replay_json_get_headers(*, referer: str) -> dict[str, str]:
    return {
        "accept": "application/json",
        "accept-language": "ru-RU,ru;q=0.9,en-US;q=0.8,en;q=0.7",
        "referer": referer,
        "sec-fetch-dest": "empty",
        "sec-fetch-mode": "cors",
        "sec-fetch-site": "same-origin",
        "user-agent": (
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/146.0.0.0 Safari/537.36"
        ),
    }


def download_episode_replay(
    session: requests.Session,
    *,
    episode_id: int,
    dest: Path,
    not_found_marker_path: Path,
    referer: str,
) -> bool:
    url = episode_replay_json_url(episode_id)
    r = session.get(
        url,
        headers=replay_json_get_headers(referer=referer),
        timeout=300.0,
    )
    time.sleep(_REQUEST_PAUSE_SEC)
    if r.status_code == 404:
        logging.warning("skip episode %s: GET %s returned HTTP 404", episode_id, url)
        write_json_atomic(
            not_found_marker_path,
            {
                "episode_id": episode_id,
                "status_code": 404,
                "captured_at_epoch_seconds": time.time(),
            },
        )
        return False
    assert r.status_code == 200, (episode_id, r.status_code, r.text[:800])
    text = r.text
    loaded = json.loads(text)
    assert isinstance(loaded, dict), (episode_id, type(loaded))
    steps = loaded["steps"]
    assert isinstance(steps, list) and len(steps) >= 2, (episode_id, type(steps), len(steps))
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(dest)
    return True


def submission_ids_round_robin_order(
    ids: list[int],
    last_started_epoch: dict[int, float],
) -> list[int]:
    assert ids
    return sorted(ids, key=lambda sid: (last_started_epoch.get(sid, 0.0), sid))


def run_iteration(
    session: requests.Session,
    *,
    competition_id: int,
    out_dir: Path,
    successful_only: bool,
    referer: str,
    build_version: str,
    xsrf_token: str | None,
    last_started_by_submission: dict[int, float],
    submission_order_state_path: Path,
) -> bool:
    payload = fetch_leaderboard_payload(
        session,
        competition_id=competition_id,
        referer=referer,
        build_version=build_version,
        xsrf_token=xsrf_token,
    )
    rows = payload["publicLeaderboard"]
    assert isinstance(rows, list), type(rows)
    nrows = len(rows)
    if nrows < _MIN_LEADERBOARD_ROWS:
        logging.warning(
            "leaderboard too short (%d rows, need %d), skip iteration",
            nrows,
            _MIN_LEADERBOARD_ROWS,
        )
        return False

    submission_ids = submission_ids_top_n_from_leaderboard_payload(
        payload,
        n=_TOP_LEADERBOARD_SUBMISSIONS,
    )
    assert submission_ids, "top submissions list empty"
    submission_order = submission_ids_round_robin_order(
        submission_ids,
        last_started_by_submission,
    )
    logging.info(
        "leaderboard rows=%d; top-%d has %d unique submissions; "
        "this pass only sid=%s (oldest last_started in round-robin)",
        nrows,
        _TOP_LEADERBOARD_SUBMISSIONS,
        len(submission_ids),
        submission_order[0],
    )

    snapshot: dict[str, object] = {
        "competition_id": competition_id,
        "captured_at_epoch_seconds": time.time(),
        "leaderboard": payload,
    }

    submission_id = submission_order[0]
    last_started_by_submission[submission_id] = time.time()
    save_submission_order_state(submission_order_state_path, last_started_by_submission)
    episodes = fetch_episodes_for_submission(
        session,
        submission_id=submission_id,
        successful_only=successful_only,
        referer=referer,
        build_version=build_version,
        xsrf_token=xsrf_token,
    )
    logging.info(
        "submission %s: %d episodes",
        submission_id,
        len(episodes),
    )
    for ep in episodes:
        eid_raw = ep["id"]
        assert isinstance(eid_raw, int), (submission_id, type(eid_raw))
        path = replay_path(out_dir, eid_raw)
        meta_path = replay_metadata_path(out_dir, eid_raw)
        not_found_path = replay_not_found_path(out_dir, eid_raw)
        if episode_bundle_is_complete(out_dir, eid_raw):
            continue
        if replay_not_found_marker_is_valid(not_found_path, eid_raw):
            logging.info(
                "skip episode %s: cached replay.json HTTP 404 marker %s",
                eid_raw,
                not_found_path,
            )
            continue
        assert "state" in ep, (submission_id, eid_raw, sorted(ep.keys()))
        state = ep["state"]
        assert isinstance(state, str), (submission_id, eid_raw, type(state))
        if state != "COMPLETED":
            logging.info(
                "skip episode %s: state=%s, replay is not ready",
                eid_raw,
                state,
            )
            continue
        logging.info(
            "refresh episode %s -> %s (missing/corrupt replay or metadata %s)",
            eid_raw,
            path,
            meta_path,
        )
        downloaded = download_episode_replay(
            session,
            episode_id=eid_raw,
            dest=path,
            not_found_marker_path=not_found_path,
            referer=referer,
        )
        if not downloaded:
            continue
        meta_path = replay_metadata_path(out_dir, eid_raw)
        write_json_atomic(meta_path, snapshot)
        logging.info("wrote leaderboard snapshot -> %s", meta_path)
    return True


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--interval-minutes",
        type=float,
        default=5.0,
        help="Пауза между итерациями и после ошибки итерации (default: 10).",
    )
    p.add_argument(
        "--competition-id",
        type=int,
        default=138420,
        help="competitionId for GetLeaderboard (default: 138420).",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/kaggle_episode_replays"),
        help="Directory for replay_<episodeId>.json and replay_<episodeId>_metadata.json.",
    )
    p.add_argument(
        "--submission-order-state",
        type=Path,
        default=None,
        help=(
            "JSON с last_started по submissionId (атомарная запись). "
            "По умолчанию: <output-dir>/kaggle_mirror_last_started_by_submission.json"
        ),
    )
    p.add_argument(
        "--cookie",
        default=None,
        help="Строка Cookie как в DevTools; иначе KAGGLE_COOKIE.",
    )
    p.add_argument(
        "--referer",
        default=None,
        help="Referer (по умолчанию orbit-wars leaderboard или KAGGLE_REFERER).",
    )
    p.add_argument(
        "--xsrf-token",
        default=None,
        help="Явный x-xsrf-token; иначе из cookie XSRF-TOKEN / CSRF-TOKEN.",
    )
    p.add_argument(
        "--kaggle-build-version",
        default=None,
        help=(
            "x-kaggle-build-version (по умолчанию из KAGGLE_BUILD_VERSION или встроенный "
            "хэш из захвата DevTools)."
        ),
    )
    p.add_argument(
        "--all-episodes",
        action="store_true",
        help="successfulOnly=false для ListEpisodes.",
    )
    p.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="DEBUG logging.",
    )
    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    cookie_str = (args.cookie or os.environ.get("KAGGLE_COOKIE", "")).strip()
    referer = (
        (args.referer or os.environ.get("KAGGLE_REFERER", "")).strip()
        or _DEFAULT_REFERER
    )
    build_ver = (
        (
            args.kaggle_build_version
            or os.environ.get("KAGGLE_BUILD_VERSION", "")
        ).strip()
        or _DEFAULT_BUILD_VERSION
    )
    xsrf = (args.xsrf_token or os.environ.get("KAGGLE_XSRF_TOKEN", "")).strip() or None
    if xsrf is None and cookie_str:
        xsrf = xsrf_from_cookie(cookie_str)
        if xsrf:
            logging.info("x-xsrf-token из Cookie")

    session = requests.Session()
    if cookie_str:
        session.headers["Cookie"] = cookie_str

    successful_only = not args.all_episodes
    interval_s = float(args.interval_minutes) * 60.0
    assert interval_s > 0, args.interval_minutes

    logging.info(
        "loop: competition_id=%s interval_min=%s out=%s "
        "successful_only=%s build=%s xsrf=%s cookie=%s",
        args.competition_id,
        args.interval_minutes,
        args.output_dir,
        successful_only,
        build_ver,
        "yes" if xsrf else "no",
        "yes" if cookie_str else "no",
    )

    submission_order_state_path = (
        args.submission_order_state
        if args.submission_order_state is not None
        else args.output_dir / _SUBMISSION_ORDER_STATE_FILENAME
    )
    last_started_by_submission = load_submission_order_state(submission_order_state_path)
    logging.info(
        "submission round-robin state: %d entries from %s",
        len(last_started_by_submission),
        submission_order_state_path,
    )

    while True:
        try:
            finished = run_iteration(
                session,
                competition_id=args.competition_id,
                out_dir=args.output_dir,
                successful_only=successful_only,
                referer=referer,
                build_version=build_ver,
                xsrf_token=xsrf,
                last_started_by_submission=last_started_by_submission,
                submission_order_state_path=submission_order_state_path,
            )
            if not finished:
                logging.info("iteration skipped (bad leaderboard), sleep anyway")
        except Exception:
            logging.exception(
                "iteration failed; retry in %.1f min (%.1f s)",
                args.interval_minutes,
                interval_s,
            )
            time.sleep(interval_s)
            continue
        logging.info("sleep %.1f s until next pass", interval_s)
        time.sleep(interval_s)


if __name__ == "__main__":
    main(sys.argv[1:])
