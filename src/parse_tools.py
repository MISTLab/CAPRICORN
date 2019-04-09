import glob
# Returns the file names of all constel files in the provided folder (assumes all .txt files are the constel files)
def get_constel_files_in_folder(folder):
    return sorted(glob.glob(folder + '/*.txt'))

