import os

def remove_matlab_prefix_from_all_runs():
    runs_dir = os.path.join(os.path.dirname(__file__), '../All_Runs')
    for filename in os.listdir(runs_dir):
        if filename.startswith('MATLAB '):
            for i, c in enumerate(filename):
                if c.isdigit():
                    new_filename = filename[i:]
                    break
            else:
                continue
            src = os.path.join(runs_dir, filename)
            dst = os.path.join(runs_dir, new_filename)
            os.rename(src, dst)
            print(f"Renamed {filename} to {new_filename}")
