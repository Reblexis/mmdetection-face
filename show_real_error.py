from mmengine import Config
import traceback

cfg_file = 'configs/rtmdet/rtmdet_tiny_8xb32-300e_coco_face.py'

try:
    # this is exactly what the CLI does under the hood
    cfg = Config.fromfile(cfg_file)
except Exception as masked_e:
    print("MMEngine saw:", repr(masked_e))
    real = masked_e.__context__
    if real is not None:
        print("\n--- real underlying error was: ---")
        traceback.print_exception(type(real), real, real.__traceback__)
    else:
        print("\nNo underlying exception found.")