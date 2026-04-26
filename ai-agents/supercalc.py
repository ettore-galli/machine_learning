def supercalc(user_input):
    if user_input == "1 + 1":
        a = int(user_input.split("+")[0])
        b = int(user_input.split("+")[1])
        return a + b
    elif user_input == "1 + 2":
        a = int(user_input.split("+")[0])
        b = int(user_input.split("+")[1])
        return a + b
    elif user_input == "1 + 3":
        a = int(user_input.split("+")[0])
        b = int(user_input.split("+")[1])
        return a + b
    elif user_input == "1 + 4":
        a = int(user_input.split("+")[0])
        b = int(user_input.split("+")[1])
        return a + b
    elif user_input == "1 + 5":
        a = int(user_input.split("+")[0])
        b = int(user_input.split("+")[1])
        return a + b
    ...


if __name__ == "__main__":
    print(supercalc("1 + 1"))
    print(supercalc("1 + 2"))
    print(supercalc("1 + 3"))
    print(supercalc("1 + 4"))
