import configparser
import datetime
from functools import partial
import glob
import logging
import multiprocessing
import os
from pathlib import Path
import sys
import time
import traceback
from types import SimpleNamespace

sys.path.append("/histoqc")

from histoqc._parser import get_argument_parser
from histoqc._pipeline import BatchedResultFile, MultiProcessingLogManager, load_pipeline, log_pipeline, move_logging_file_handler, setup_logging, setup_plotting_backend
from histoqc._worker import worker, worker_error, worker_setup, worker_success
from histoqc.data import managed_pkg_data


parser = get_argument_parser()


def main(options: dict):
    """
    main entry point for simplified histoqc pipelines (dict 기반)

    options 예시:
    {
        "config": "/workspace/practice/config.ini",
        "outdir": "/workspace/practice/output",
        "batch": 4,
        "force": True,
        "seed": 42,
        "input_dir": "/workspace/practice/input",
        # 선택값:
        # "nprocesses": 4,
        # "debug": False,
    }
    """

    # --- 필수 인자 체크 ------------------------------------------------------
    required_keys = ["config", "outdir", "input_dir"]
    for k in required_keys:
        if k not in options or options[k] is None:
            raise ValueError(f"'{k}' 값이 필요합니다. options['{k}']를 설정해주세요.")

    # --- dict → args(SimpleNamespace) 변환 -----------------------------------
    args = SimpleNamespace(
        # 필수
        config=options["config"],
        outdir=options["outdir"],
        input_dir=options["input_dir"],
        # 선택 / 기본값
        batch=int(options.get("batch", 4)),
        force=bool(options.get("force", False)),
        seed=int(options.get("seed", 42)),
        nprocesses=int(options.get("nprocesses", 1)),
        debug=bool(options.get("debug", False)),
    )

    print("args:", args)

    # --- multiprocessing and logging setup -----------------------------------
    setup_logging(capture_warnings=True, filter_warnings="ignore")
    mpm = multiprocessing.Manager()
    lm = MultiProcessingLogManager("histoqc", manager=mpm)

    # --- parse the pipeline configuration ------------------------------------
    config = configparser.ConfigParser()
    config.read(args.config)

    # --- load the process queue (error early) --------------------------------
    _steps = log_pipeline(config, log_manager=lm)
    process_queue = load_pipeline(config)

    # --- create output directory and move log --------------------------------
    args.outdir = os.path.expanduser(args.outdir)
    os.makedirs(args.outdir, exist_ok=True)
    move_logging_file_handler(logging.getLogger(), args.outdir, args.debug)

    if BatchedResultFile.results_in_path(args.outdir):
        if args.force:
            lm.logger.info("Previous run detected....overwriting (force=True)")
        else:
            lm.logger.info("Previous run detected....skipping completed (force=False)")

    results = BatchedResultFile(
        args.outdir,
        manager=mpm,
        batch_size=args.batch,
        force_overwrite=args.force,
    )

    # --- document configuration in results -----------------------------------
    # command_line_args 는 dict 기반으로 문자열 만들어서 기록
    cmdline_repr = " ".join(f"{k}={v}" for k, v in options.items())

    results.add_header(f"start_time:\t{datetime.datetime.now()}")
    results.add_header(f"pipeline:\t{' '.join(_steps)}")
    results.add_header(f"outdir:\t{os.path.realpath(args.outdir)}")
    results.add_header(
        f"config_file:\t{os.path.realpath(args.config) if args.config is not None else 'default'}"
    )
    results.add_header(f"command_line_args:\t{cmdline_repr}")

    # --- receive input file list (간소화: input_dir 안의 .svs 파일만) ---------
    input_dir = args.input_dir
    files = []
    for input_file in os.listdir(input_dir):
        if input_file.endswith("svs"):
            file_path = Path(input_dir) / input_file
            files.append(str(file_path))

    lm.logger.info("-" * 80)
    num_files = len(files)
    lm.logger.info(f"Number of files detected by pattern:\t{num_files}")

    # --- start worker processes ----------------------------------------------
    _shared_state = {
        "process_queue": process_queue,
        "config": config,
        "outdir": args.outdir,
        "log_manager": lm,
        "lock": mpm.Lock(),
        "shared_dict": mpm.dict(),
        "num_files": num_files,
        "force": args.force,
        "seed": args.seed,
    }
    failed = mpm.list()

    if args.nprocesses > 1:
        with lm.logger_thread():
            lm.logger.info(f"Using {args.nprocesses} worker processes")
            with multiprocessing.Pool(
                processes=args.nprocesses,
                initializer=worker_setup,
                initargs=(config,),
            ) as pool:
                try:
                    for idx, file_name in enumerate(files):
                        pool.apply_async(
                            func=worker,
                            args=(idx, file_name),
                            kwds=_shared_state,
                            callback=partial(worker_success, result_file=results),
                            error_callback=partial(worker_error, failed=failed),
                        )
                finally:
                    pool.close()
                    pool.join()
    else:
        print("여기")
        # 싱글 프로세스 모드
        for idx, file_name in enumerate(files):
            start = time.perf_counter()
            print(idx, file_name)
            try:
                _success = worker(idx, file_name, **_shared_state)
            except Exception as exc:
                traceback.print_exc()
                worker_error(exc, failed)
                continue
            else:
                worker_success(_success, results)

            print("duration: ", time.perf_counter() - start, "\n\n")

if __name__ == "__main__":
    argv = {
        "config": "/histoqc/practice/config.ini",
        "outdir": "/histoqc/practice/output",
        "batch": 4,
        "force": True,
        "seed": 42,
        "input_dir": "/histoqc/practice/input",
        "input_pattern": [],
        "nprocesses": 1
    }

    main(argv)