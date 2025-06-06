import sys
import sysconfig

print(f"Python Version: {sys.version}")
status = sysconfig.get_config_var("Py_GIL_DISABLED")
if status is None:
    print("GIL cannot be disabled")
elif status == 0:
    print("GIL is enabled")
elif status == 1:
    print("GIL is disabled")
