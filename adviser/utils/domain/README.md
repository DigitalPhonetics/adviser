# Purpose:
The domain classes define ways to interact with a data source and an ontology in order to carry out a task-oriented dialog in a specific domain.

# Description of Files:
* `domain.py`: Defines a parent class for domains, creating a common interface for domain classes which should all have a domain name and a way to find entities
* `jsonlookupdomain.py`: Defines a domain class which takes in a JSON file as an ontology description and a SQLite database as a datasource
* `lookupdomain.py`: Defines a slighly more concrete interface for a domain object with method interfaces for reading an ontology