import sys, getopt, time, statistics
from datetime import timedelta




def progress_bar(progress, total, prefix="Progress:", sufix="Complete", length=40, clear=False):
    progress += 1
    try:
        if not hasattr(progress_bar, "begin_time") or clear==True:
            progress_bar.begin_time = time.time()
            progress_bar.current_time = time.time()
            progress_bar.all_times = []

        progress_bar.prev_time = progress_bar.current_time
        progress_bar.current_time = time.time() 
        progress_bar.step_time = progress_bar.current_time - progress_bar.prev_time
        progress_bar.all_times.append(progress_bar.step_time)
        progress_bar.mean_step_time = statistics.mean(progress_bar.all_times[-100:])
        progress_bar.elapsed_time = progress_bar.current_time - progress_bar.begin_time

        percent = 100 * (progress/ float(total))
        filled_length = int(round(length * (progress/ float(total))))
        bar = '█' * filled_length + '_' * (length - filled_length)
        i_of = f'{progress:>4d}/{total}'
        print_bar = f"\r{prefix} |{bar}| {i_of} {percent:>6.2f}% {sufix}"

        time_left = progress_bar.mean_step_time * (total-1 - progress)
        time_left = time.strftime("%H:%M:%S", time.gmtime(time_left))
        elapsed_time = time.strftime("%H:%M:%S", time.gmtime(progress_bar.elapsed_time))
        print_time = f" - ETA: {time_left} | Elapsed: {elapsed_time} | MPT: {progress_bar.mean_step_time:.2f}"

        print(print_bar + print_time, end="\r")
        if progress == total:
            print(print_bar + print_time, end='\n')

    except KeyboardInterrupt:
        print('caught')