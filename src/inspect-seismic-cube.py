import numpy as np
import util.filehandler as fh
import util.plotting as draw

def main():
    cube_path = r"C:\Users\artemiy\Documents\University\Диплом\х Фактические данные\Куб, интерпретация, ГИС\Куб (2015)\Vankorskaya_s_p_5_03-04_Migrirovannyiy_PreStack.sgy"
    traces, il, xl = fh.read_sgy_selective(cube_path)
    print(traces[:100,:100,:100])
    # draw.plot_3d_seismic(traces)

if __name__ == "__main__":
    main()
