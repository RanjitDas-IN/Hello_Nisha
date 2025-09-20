content = []
with open(r"Hello_Nisha/txt", "r") as file:
   for i in file:
       content.append(i.strip())


data = "\n".join(f'{line}|' for line in content)
print(data)


with open("Hello_Nisha/txt", "w") as file:
    file.write(data) 