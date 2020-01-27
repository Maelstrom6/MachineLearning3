import os

directory = "C:\Program Files (x86)\Steam\steamapps\common\Counter-Strike Global Offensive"
string_to_find = "getout"

for r, d, f in os.walk(directory):
    for file in f:
            file_name = os.path.join(r, file)
            try:
                file = open(file_name, "r", errors="replace")
                file_text = file.read()
                if file_text.find(string_to_find) != -1:
                    print(file_name)
                file.close()
            except UnicodeDecodeError:
                print("wrong encoding for", file_name)


# for r, d, f in os.walk(directory):
#     for file in f:
#         if ".txt" not in file:
#             file_name = os.path.join(r, file)
#             if "radio" in file_name:
#                 print(file_name)


