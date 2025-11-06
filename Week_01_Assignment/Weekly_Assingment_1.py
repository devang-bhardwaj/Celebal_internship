rows = 7

print("Lower Triangle:")
for x in range(1, rows + 1):
    result = ""
    count = 0
    while count < x:
        result += "*"
        count += 1
    print(result)

print("")

print("Upper Triangle:")
num = rows
for y in range(rows):
    output = ""
    temp = 0
    while temp < num:
        output = output + "*"
        temp = temp + 1
    print(output)
    num = num - 1

print("")

print("Pyramid:")
for z in range(rows):
    blank_spaces = ""
    for space_count in range(rows - z - 1):
        blank_spaces = blank_spaces + " "
    
    star_part = ""
    star_counter = 0
    while star_counter <= z:
        star_part = star_part + "*"
        star_counter += 1
    
    extra_stars = ""
    if z > 0:
        for additional in range(z):
            extra_stars += "*"
    
    final_line = blank_spaces + star_part + extra_stars
    print(final_line)