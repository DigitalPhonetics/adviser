# Purpose

The Mensa API example shows you how to create a task-oriented dialogue system without a database as information source. Instead of looking up entities in a database, the Mensa menus are dynamically extracted at runtime.

# Files

`domain.py`: The code for the domain which performs the call to the Mensa website <br>
`nlu.py`: The code for the Mensa-specific natural language understanding <br>
`parser.py`: The code for parsing the Mensa (HTML) website to extract the required information and convert it to structured knowledge
