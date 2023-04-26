from services.nlu.nlu import HandcraftedNLU
from utils.domain.jsonlookupdomain import JSONLookupDomain


# 1st step: compile your .nlu file into regular expressions:
#   python3 tools/regextemplates/gen_regexes.py superhero superhero
# 2nd step: test your NLU by running
#   python3 test_nlu.py

if __name__ == "__main__":
    # create a NLU object for the superhero domain
    print("NLU TEST")
    nlu = HandcraftedNLU(domain=JSONLookupDomain(name='superhero', # domain name
                                                json_ontology_file='resources/ontologies/superhero.json', # path to ontology, relative to adviser folder
                                                sqllite_db_file='resources/databases/superhero.db', # path to sqlite database, relative to adviser folder
                                                display_name='Superhero Domain')) # human-readable name of the domain
    user_input = None
    while not user_input == "exit":
        # read in next test string from terminal
        user_input = input(">>")
        # extract user acts  from this string
        acts = nlu.extract_user_acts(user_input)
        # print the extracted user acts
        print(acts)
        print("=================")
    
    