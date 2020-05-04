# Purpose:
This folder contains regex files, and templates for making regex files, which map natural language user input to user actions.

* `.nlu` files represent NLU templates which can be used to automatically generate regex files
* `.json` files represent the complete regexes used by the NLU
* three types of `.json` files:
  * **General**
     * Contain general (non-domain specific) regexes for English and German (based on end extension)
  * **InformRules**
    * Contain regexes for user informs
  * **RequestRules** 
    * Contain regexes for user requests
* File names are in the following formats:
  * For english files:
    * `{domain_name}{Type}{Language}.json`
    * If no language is specified, the file is in English