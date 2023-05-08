from examples.ml_nlu.MLNLU import MLNLU

# This will download a pre-trained NLU (MultiWoz domain: Restaurants, Hotels, Transportation, ...)
# More info on the domains: https://github.com/budzianowski/multiwoz/tree/master/data/MultiWOZ_2.2

if __name__ == "__main__":
    # create a NLU object for the superhero domain
    print("ML-based NLU TEST\nType 'exit' to end dialog.")
    nlu = MLNLU(domain="restaurants") # human-readable name of the domain
    user_input = None
    while not user_input == "exit":
        # read in next test string from terminal
        user_input = input(">>")
        # extract user acts  from this string
        acts = nlu.extract_user_acts(user_input)
        # print the extracted user acts
        print(acts)
        print("=================")
    
    