#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import requests

from kaggle_leaderboard_episode_mirror import (
    _DEFAULT_BUILD_VERSION,
    api_headers,
    xsrf_from_cookie,
    write_json_atomic,
)

_URL_LIST_EPISODES = (
    "https://www.kaggle.com/api/i/competitions.EpisodeService/ListEpisodes"
)
_URL_AGENT_LOGS_JSON = (
    "https://www.kaggle.com/competitions/episodes/{episode_id}/agents/{agent_id}/logs.json"
)
_DEFAULT_AGENT_IDS = "0,1,2,3"
_DEFAULT_REQUEST_PAUSE_SEC = 2.0


def _default_referer(submission_id: int) -> str:
    return f"https://www.kaggle.com/competitions/orbit-wars/submissions?submissionId={submission_id}"


def _parse_agent_ids(raw: str) -> list[int]:
    pieces = [p.strip() for p in raw.split(",")]
    assert pieces, raw
    out: list[int] = []
    seen: set[int] = set()
    for piece in pieces:
        assert piece, raw
        agent_id = int(piece)
        assert agent_id >= 0, agent_id
        assert agent_id not in seen, raw
        seen.add(agent_id)
        out.append(agent_id)
    assert out, raw
    return out


def agent_logs_json_url(*, episode_id: int, agent_id: int) -> str:
    return _URL_AGENT_LOGS_JSON.format(episode_id=episode_id, agent_id=agent_id)


def agent_logs_path(out_dir: Path, *, episode_id: int, agent_id: int) -> Path:
    return out_dir / f"episode_{episode_id}_agent_{agent_id}_logs.json"


def agent_logs_metadata_path(out_dir: Path, *, episode_id: int, agent_id: int) -> Path:
    return out_dir / f"episode_{episode_id}_agent_{agent_id}_logs_metadata.json"


def agent_logs_not_found_path(out_dir: Path, *, episode_id: int) -> Path:
    return out_dir / f"episode_{episode_id}_agent_logs_not_found.json"


def agent_logs_file_is_valid(path: Path) -> bool:
    if not path.is_file():
        return False
    text = path.read_text(encoding="utf-8")
    raw = json.loads(text)
    return isinstance(raw, dict) or isinstance(raw, list)


def agent_logs_metadata_is_valid(
    path: Path,
    *,
    submission_id: int,
    episode_id: int,
    agent_id: int,
) -> bool:
    if not path.is_file():
        return False
    text = path.read_text(encoding="utf-8")
    raw = json.loads(text)
    assert isinstance(raw, dict), type(raw)
    assert raw["submission_id"] == submission_id, raw
    assert raw["episode_id"] == episode_id, raw
    assert raw["agent_id"] == agent_id, raw
    assert raw["status_code"] == 200, raw
    ts = raw["captured_at_epoch_seconds"]
    assert type(ts) is int or type(ts) is float, type(ts)
    return True


def existing_downloaded_agent_id(
    out_dir: Path,
    *,
    submission_id: int,
    episode_id: int,
    agent_ids: list[int],
) -> int | None:
    for agent_id in agent_ids:
        logs_path = agent_logs_path(out_dir, episode_id=episode_id, agent_id=agent_id)
        metadata_path = agent_logs_metadata_path(out_dir, episode_id=episode_id, agent_id=agent_id)
        if agent_logs_file_is_valid(logs_path) and agent_logs_metadata_is_valid(
            metadata_path,
            submission_id=submission_id,
            episode_id=episode_id,
            agent_id=agent_id,
        ):
            return agent_id
    return None


def agent_logs_not_found_marker_is_valid(
    path: Path,
    *,
    submission_id: int,
    episode_id: int,
    agent_ids: list[int],
) -> bool:
    if not path.is_file():
        return False
    text = path.read_text(encoding="utf-8")
    raw = json.loads(text)
    assert isinstance(raw, dict), type(raw)
    assert raw["submission_id"] == submission_id, raw
    assert raw["episode_id"] == episode_id, raw
    assert raw["agent_ids"] == agent_ids, raw
    ts = raw["captured_at_epoch_seconds"]
    assert type(ts) is int or type(ts) is float, type(ts)
    return True


def logs_json_get_headers(
    *,
    referer: str,
    build_version: str,
    xsrf_token: str | None,
) -> dict[str, str]:
    headers: dict[str, str] = {
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
        "x-kaggle-build-version": build_version,
    }
    if xsrf_token is not None:
        headers["x-xsrf-token"] = xsrf_token
    return headers


def fetch_episodes_for_submission(
    session: requests.Session,
    *,
    submission_id: int,
    successful_only: bool,
    referer: str,
    build_version: str,
    xsrf_token: str | None,
    request_pause_sec: float,
) -> list[dict[str, object]]:
    response = session.post(
        _URL_LIST_EPISODES,
        json={
            "ids": [],
            "submissionId": submission_id,
            "successfulOnly": successful_only,
        },
        headers=api_headers(
            referer=referer,
            build_version=build_version,
            xsrf_token=xsrf_token,
        ),
        timeout=120.0,
    )
    time.sleep(request_pause_sec)
    if response.status_code != 200:
        content_type = response.headers.get("Content-Type", "")
        assert False, (
            f"HTTP {response.status_code} {_URL_LIST_EPISODES!r} "
            f"content-type={content_type!r} body[:2000]={response.text[:2000]!r}"
        )
    obj = response.json()
    assert isinstance(obj, dict), type(obj)
    if "result" in obj:
        payload = obj["result"]
        assert isinstance(payload, dict), type(payload)
    else:
        payload = obj
    episodes = payload["episodes"]
    assert isinstance(episodes, list), type(episodes)
    out: list[dict[str, object]] = []
    for episode in episodes:
        assert isinstance(episode, dict), type(episode)
        assert "id" in episode, sorted(episode.keys())
        out.append(episode)
    return out


def download_first_available_agent_logs(
    session: requests.Session,
    *,
    submission_id: int,
    episode_id: int,
    agent_ids: list[int],
    out_dir: Path,
    referer: str,
    build_version: str,
    xsrf_token: str | None,
    request_pause_sec: float,
) -> int | None:
    for agent_id in agent_ids:
        url = agent_logs_json_url(episode_id=episode_id, agent_id=agent_id)
        logging.info("try episode=%s agent=%s url=%s", episode_id, agent_id, url)
        response = session.get(
            url,
            headers=logs_json_get_headers(
                referer=referer,
                build_version=build_version,
                xsrf_token=xsrf_token,
            ),
            timeout=300.0,
        )
        time.sleep(request_pause_sec)
        if response.status_code in (400, 403, 404):
            logging.info(
                "agent=%s unavailable: HTTP %s",
                agent_id,
                response.status_code,
            )
            continue
        if response.status_code != 200:
            content_type = response.headers.get("Content-Type", "")
            assert False, (
                f"HTTP {response.status_code} {url!r} "
                f"content-type={content_type!r} body[:2000]={response.text[:2000]!r}"
            )

        loaded = json.loads(response.text)
        assert isinstance(loaded, dict) or isinstance(loaded, list), (agent_id, type(loaded))

        dest = agent_logs_path(out_dir, episode_id=episode_id, agent_id=agent_id)
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(dest.suffix + ".tmp")
        tmp.write_text(response.text, encoding="utf-8")
        tmp.replace(dest)

        write_json_atomic(
            agent_logs_metadata_path(out_dir, episode_id=episode_id, agent_id=agent_id),
            {
                "submission_id": submission_id,
                "episode_id": episode_id,
                "agent_id": agent_id,
                "url": url,
                "status_code": response.status_code,
                "captured_at_epoch_seconds": time.time(),
            },
        )
        logging.info("wrote agent logs -> %s", dest)
        return agent_id

    return None


def write_agent_logs_not_found_marker(
    out_dir: Path,
    *,
    submission_id: int,
    episode_id: int,
    agent_ids: list[int],
) -> None:
    write_json_atomic(
        agent_logs_not_found_path(out_dir, episode_id=episode_id),
        {
            "submission_id": submission_id,
            "episode_id": episode_id,
            "agent_ids": agent_ids,
            "captured_at_epoch_seconds": time.time(),
        },
    )


def download_submission_agent_logs(
    session: requests.Session,
    *,
    submission_id: int,
    agent_ids: list[int],
    out_dir: Path,
    successful_only: bool,
    referer: str,
    build_version: str,
    xsrf_token: str | None,
    request_pause_sec: float,
) -> None:
    episodes = fetch_episodes_for_submission(
        session,
        submission_id=submission_id,
        successful_only=successful_only,
        referer=referer,
        build_version=build_version,
        xsrf_token=xsrf_token,
        request_pause_sec=request_pause_sec,
    )
    logging.info("submission=%s episodes=%s", submission_id, len(episodes))
    assert episodes, submission_id

    for episode in episodes:
        episode_id_raw = episode["id"]
        assert isinstance(episode_id_raw, int), (submission_id, type(episode_id_raw))
        state_raw = episode["state"]
        assert isinstance(state_raw, str), (submission_id, episode_id_raw, type(state_raw))
        if state_raw != "COMPLETED":
            logging.info(
                "skip episode=%s state=%s; logs are not ready",
                episode_id_raw,
                state_raw,
            )
            continue
        existing_agent_id = existing_downloaded_agent_id(
            out_dir,
            submission_id=submission_id,
            episode_id=episode_id_raw,
            agent_ids=agent_ids,
        )
        if existing_agent_id is not None:
            logging.info(
                "skip episode=%s: cached logs already exist for agent=%s",
                episode_id_raw,
                existing_agent_id,
            )
            continue
        not_found_path = agent_logs_not_found_path(out_dir, episode_id=episode_id_raw)
        if agent_logs_not_found_marker_is_valid(
            not_found_path,
            submission_id=submission_id,
            episode_id=episode_id_raw,
            agent_ids=agent_ids,
        ):
            logging.info(
                "skip episode=%s: cached no-agent-logs marker %s",
                episode_id_raw,
                not_found_path,
            )
            continue
        found_agent_id = download_first_available_agent_logs(
            session,
            submission_id=submission_id,
            episode_id=episode_id_raw,
            agent_ids=agent_ids,
            out_dir=out_dir,
            referer=referer,
            build_version=build_version,
            xsrf_token=xsrf_token,
            request_pause_sec=request_pause_sec,
        )
        if found_agent_id is None:
            write_agent_logs_not_found_marker(
                out_dir,
                submission_id=submission_id,
                episode_id=episode_id_raw,
                agent_ids=agent_ids,
            )
            logging.info("wrote no-agent-logs marker for episode=%s", episode_id_raw)
            continue
        logging.info(
            "downloaded submission=%s episode=%s agent=%s",
            submission_id,
            episode_id_raw,
            found_agent_id,
        )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Download first accessible Orbit Wars Kaggle agent logs.json for each submission episode.",
    )
    parser.add_argument("--submission-id", type=int, required=True)
    parser.add_argument(
        "--agent-ids",
        default=_DEFAULT_AGENT_IDS,
        help=f"Comma-separated agent ids to try in order (default: {_DEFAULT_AGENT_IDS}).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/kaggle_agent_logs"),
        help="Directory for episode_<episodeId>_agent_<agentId>_logs.json.",
    )
    parser.add_argument(
        "--all-episodes",
        action="store_true",
        help="successfulOnly=false for ListEpisodes.",
    )
    parser.add_argument(
        "--request-pause-seconds",
        type=float,
        default=_DEFAULT_REQUEST_PAUSE_SEC,
        help=f"Pause after each Kaggle request in this script (default: {_DEFAULT_REQUEST_PAUSE_SEC}).",
    )
    parser.add_argument(
        "--cookie",
        default=None,
        help="Cookie header copied from DevTools; otherwise KAGGLE_COOKIE.",
    )
    parser.add_argument(
        "--referer",
        default=None,
        help="Referer; default is the Orbit Wars submission page, or KAGGLE_REFERER.",
    )
    parser.add_argument(
        "--xsrf-token",
        default=None,
        help="Explicit x-xsrf-token; otherwise KAGGLE_XSRF_TOKEN or Cookie XSRF-TOKEN / CSRF-TOKEN.",
    )
    parser.add_argument(
        "--kaggle-build-version",
        default=None,
        help="x-kaggle-build-version; otherwise KAGGLE_BUILD_VERSION or captured default.",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="DEBUG logging.")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    submission_id = int(args.submission_id)
    assert submission_id > 0, submission_id
    agent_ids = _parse_agent_ids(str(args.agent_ids))
    request_pause_sec = float(args.request_pause_seconds)
    assert request_pause_sec >= 0.0, request_pause_sec

    cookie_str = (args.cookie or os.environ.get("KAGGLE_COOKIE", "")).strip()
    referer = (
        (args.referer or os.environ.get("KAGGLE_REFERER", "")).strip()
        or _default_referer(submission_id)
    )
    build_version = (
        (args.kaggle_build_version or os.environ.get("KAGGLE_BUILD_VERSION", "")).strip()
        or _DEFAULT_BUILD_VERSION
    )
    xsrf_token = (args.xsrf_token or os.environ.get("KAGGLE_XSRF_TOKEN", "")).strip() or None
    if xsrf_token is None and cookie_str:
        xsrf_token = xsrf_from_cookie(cookie_str)
        if xsrf_token is not None:
            logging.info("x-xsrf-token from Cookie")

    session = requests.Session()
    if cookie_str:
        session.headers["Cookie"] = cookie_str

    successful_only = not args.all_episodes
    download_submission_agent_logs(
        session,
        submission_id=submission_id,
        agent_ids=agent_ids,
        out_dir=args.output_dir,
        successful_only=successful_only,
        referer=referer,
        build_version=build_version,
        xsrf_token=xsrf_token,
        request_pause_sec=request_pause_sec,
    )
    logging.info("done: submission=%s", submission_id)


if __name__ == "__main__":
    main(sys.argv[1:])
