import os
import os.path as osp
import importlib
import inspect
from .base_task import BaseTask



def create_task(cfg, log):


    filenames = os.listdir(osp.dirname(__file__))
    # 过滤掉 __init__.py 和名字里包含 "备份" 的文件
    filenames = filter(
        lambda x: x.endswith('.py') and x != '__init__.py' and "备份" not in x,
        filenames
    )

    type2task = dict()
    for filename in filenames:
        module = importlib.import_module('tasks.%s' % filename[:-3])
        #importlib.reload(module)
        clsmembers = inspect.getmembers(module, inspect.isclass)
        for clsmember in clsmembers:
            cls = clsmember[1]
            if issubclass(cls, BaseTask) and cls is not BaseTask:
                print(f"[DEBUG] Found task class {cls.__name__} in {filename}")
                type2task[clsmember[0]] = cls

        # print(f"[DEBUG] Inspecting module {module}")
        # print("[DEBUG] Available task classes:", list(type2task.keys()))
        # for name, cls in inspect.getmembers(module, inspect.isclass):
        #     print("   ", name, "->", cls)
        #     print("Class in module:", name, "is subclass of BASETASK_MAIN?", issubclass(cls, BaseTask))

    chosen_cls = type2task[cfg.task_type]
    print(f"[INFO] Using task class {chosen_cls.__name__} from {chosen_cls.__module__}")
    return chosen_cls(cfg, log)