import glob
import json


def open_and_load(file_name):
    with open(file_name) as source:
        return json.load(source)


tc_files = glob.glob("/data/11/nict_dl_data/json_ontonotes/ch*")
bc_files = [
    file_name
    for file_name in glob.glob("/data/11/nict_dl_data/json_ontonotes/*")
    if file_name not in tc_files
]
ami_files = glob.glob("/data/11/nict_dl_data/json_ami_corpus/*")
tcs = map(open_and_load, tc_files)
bcs = map(open_and_load, bc_files)
amis = map(open_and_load, ami_files)
print("# TC")
print(sum(map(len, tcs)))
print("# BC")
print(sum(map(len, bcs)))
print("# AMI")
print(sum(map(len, amis)))
