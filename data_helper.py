from pathlib import Path
import shutil


def main():
    directory_name = "Data/All/"
    directory = Path(directory_name)

    # Get all file names in the directory
    file_names = [directory_name+f.name for f in directory.iterdir() if f.is_file()]
    print(file_names[0].split())
    
    months_after_may = {"Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"}

    files_after_may = [file for file in file_names if file.split(' ')[4] in months_after_may]

    files_may_and_before = [file for file in file_names if file.split(' ')[4] not in months_after_may]

    for fn in files_after_may:
        real_fn = fn.split('/')[-1]
        shutil.copy("Data/All/"+real_fn, "Data/After_May/"+real_fn)

    for fn in files_may_and_before:
        real_fn = fn.split('/')[-1]
        shutil.copy("Data/All/"+real_fn, "Data/May_And_Before/"+real_fn)

if __name__ == "__main__":
    main()