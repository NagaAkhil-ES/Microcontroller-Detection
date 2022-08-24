import time
from functools import wraps

def log_time_s(func):
    """function to use as decorator"""
    @wraps(func)
    def calculate_run_time(*args, **kwargs):     # if function takes any arguments, can be added like this.
        start_time=time.time()  # storing time before function execution
        return_value = func(*args, **kwargs)
        run_time = (time.time()-start_time)    # calc run time in sec after function execution
        print(f"{func.__name__} function time cost: {run_time:.2f}s")
        return return_value
    return calculate_run_time

def log_time_ms(func):
    """function to use as decorator"""
    @wraps(func)
    def calculate_run_time(*args, **kwargs):     # if function takes any arguments, can be added like this.
        start_time=time.time()  # storing time before function execution
        return_value = func(*args, **kwargs)
        run_time = (time.time()-start_time)    # calc run time in sec after function execution
        print(f"{func.__name__} function time cost: {run_time*999:.2f}ms")
        return return_value
    return calculate_run_time

def get_run_time(start_time, unit="sec"):
    """unit = "sec" | "min" | "hr" """
    end_time=time.time()
    run_time=round(end_time-start_time)  # calc runtime in sec
    if unit == "min":
        run_time=round((run_time/60), 2)
    elif unit == "hr":
        run_time=round((run_time/3600), 2)
    return run_time

if __name__ == "__main__":
    @log_time_ms
    def delay():
        time.sleep(1.5)
    delay()