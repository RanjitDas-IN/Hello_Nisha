content = []
with open(r"/home/ranjit/Desktop/projects/Hello_Nisha/data/single_wake_up_dataset.csv", "r") as file:
   for i in file:
       content.append(i.strip())


data = "\n".join(f'{line}|' for line in content)
print(data)

with open("/home/ranjit/Desktop/projects/Hello_Nisha/data/temp.csv", "w") as file:
    file.write(data) 