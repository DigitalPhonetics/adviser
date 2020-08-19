############################################################################################
#
# Copyright 2020, University of Stuttgart: Institute for Natural Language Processing (IMS)
#
# This file is part of Adviser.
# Adviser is free software: you can redistribute it and/or modify'
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3.
#
# Adviser is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Adviser.  If not, see <https://www.gnu.org/licenses/>.
#
############################################################################################

from typing import Tuple, List, Dict
import datetime
import re
from lxml import html
import requests
from enum import Enum


class ParseDateError(Exception):
	"""This exception is raised when the date cannot be parsed."""
	pass


class Location(Enum):
	"""This enum provides the possible mensa locations."""

	STUTTGART_MITTE = 3
	MUSIKHOCHSCHULE = 4
	KUNSTAKADEMIE = 7
	STUTTGART_VAIHINGEN = 2
	LUDWIGSBURG = 1
	FLANDERNSTRASSE = 6
	ESSLINGEN_STADTMITTE = 9

class Allergen(Enum):
	"""This enum provides the allergens used in the mensa menu."""

	Egg = 'Ei'
	Peanut = 'En'
	Fish = 'Fi'
	Wheat = 'GlW'
	Spelt = 'GlD'
	KhorasanWheat = 'GlKW'
	Rye = 'GlR'
	Barley = 'GlG'
	Millet = 'GlH'
	Shellfishes = 'Kr'
	Lactose = 'La'
	Lupin = 'Lu'
	Almonds = 'NuM'
	Hazelnuts = 'NuH'
	Walnuts = 'NuW'
	Cashews = 'NuC'
	Pecan = 'NuPe'
	BrazilNut = 'NuPa'
	Pistachio = 'NuPi'
	Macadamia = 'NuMa'
	Sesame = 'Se'
	Mustard = 'Sf'
	Celery = 'Sl'
	Soy = 'So'
	Sulfite = 'Sw'
	Mollusca = 'Wt'

class DishType(Enum):
	"""This enum provides the dish types used in the mensa menu."""

	Starter = 'starter'
	Buffet = 'buffet'
	MainDish = 'main_dish'
	SideDish = 'side_dish'
	Dessert = 'dessert'

	@staticmethod
	def from_website_name(website_name: str) -> 'DishType':
		"""Converts the type as listed on the website into the type used in the dialog system.
		
		Args:
			website_name: The name as used in the response to the POST request.

		Returns:
			The corresponding enum member.
		
		"""


		if website_name == 'STARTER':
			return DishType.Starter
		elif website_name == 'BUFFET':
			return DishType.Buffet
		elif website_name == 'MAIN DISH':
			return DishType.MainDish
		elif website_name == 'SIDE DISH':
			return DishType.SideDish
		elif website_name == 'DESSERT':
			return DishType.Dessert

class Meal():
	def __init__(self, name: str, day: str, prices: Tuple[float], price_quantity: str,\
		allergens:List[Allergen], vegan: bool, vegetarian: bool, fish: bool, pork: bool,\
			dish_type: DishType):
		"""The class for a  meal consisting of a name and several properties (slot-value pairs).
		
		Args:
			name: The name of the meal.
			day: The day on which the meal is offered.
			prices: The price for students and guests.
			price_quantity: The unit for which the price is valid.
			allergens: The allergens of this meal.
			vegan: Whether the meal is vegan or not.
			vegetarian: Whether the meal is vegetarian or not.
			fish: Whether the meal contains fish or not.
			pork: Whether the meal contains pork or not.
			dish_type: The type of the dish. (Starter, Buffet, Main Dish, Side Dish or Buffet)

		"""
		self.name = name
		self.day = day
		self.prices = prices
		self.price_quantity = price_quantity
		self.allergens = allergens
		self.vegan = vegan
		self.vegetarian = vegetarian
		self.fish = fish
		self.pork = pork
		self.dish_type = dish_type


	def as_dict(self) -> Dict[str, str]:
		"""A dict representation of the meal."""

		return {
			'name': self.name,
			'day': self.day,
			'type': self.dish_type.value,
			'price': str(self.prices[0]),
			'allergens': ', '.join([allergen.value for allergen in self.allergens]) if\
				self.allergens is not None else 'none',
			'vegan': str(self.vegan).lower(),
			'vegetarian': str(self.vegetarian).lower(),
			'fish': str(self.fish).lower(),
			'pork': str(self.pork).lower()
			}

	def __str__(self) -> str:
		"""The string representation of the meal."""

		return (f"Meal(name={self.name}, day={self.day}, prices={self.prices},\
			price_quantity={self.price_quantity}, "
			f"allergens={self.allergens}, vegan={self.vegan}, vegetarian={self.vegetarian}, "
			f"fish={self.fish}, pork={self.pork}, dish_type={self.dish_type})")

	def __repr__(self) -> str:
		"""The string representation of the meal."""
		
		return str(self)


class MensaParser():
	def __init__(self, cache: bool = True):
		"""
		The class to issue post requests and parse the response. Will also take care of caching the
		parser's results.

		Args:
			cache (bool): Whether to cache results or not.

		"""
		
		#: dict of str: storgae to cache parsed meals
		self.storage = {}
		self.cache = cache

	def _parse(self, date: datetime.datetime) -> List[Meal]:
		"""
		Issues a request for the given date. The response will be parsed and a list of meals
		returned.

        Args:
            date: The date for which the data will be parsed.

        Returns:
            :obj:`list` of Meal: List of parsed meals

        """

		date_str = date.strftime('%Y-%m-%d')
		date_next_week_str = (date + datetime.timedelta(days=7)).strftime('%Y-%m-%d')

		data = {
		'func': 'make_spl',
		# currently we stick with only one mensa location
		'locId': Location.STUTTGART_VAIHINGEN.value,
		'date': date_str,
		'lang': 'en',
		'startThisWeek': date_str,
		'startNextWeek': date_next_week_str
		}

		# issue post request
		response = requests.post('https://sws2.maxmanager.xyz/inc/ajax-php_konnektor.inc.php',\
			headers={}, cookies={}, data=data)
		tree = html.fromstring(response.content.decode(response.encoding))
		meals = [self._parse_meal(meal, date.strftime('%A')) for meal in\
			tree.xpath('//div[contains(@class, "splMeal")]')]

		if self.storage:
			self.storage[date] = meals

		return meals

	def _parse_meal(self, meal: html.HtmlElement, day: str) -> Meal:
		"""Parse all necessary properties of a meal from html.
		
		Args:
			meal: The html.HtmlElement which will be parsed.
			day: The day for which this meal is valid.
		"""

		# name
		name = meal.xpath('./div[1]/span/text()')[0].strip()
		# price & quantity
		div_price =\
			meal.xpath('./div[4]/div[1]/text()[preceding-sibling::br or following-sibling::br]')
		prices = re.search(r'(\d*,\d*).*?(\d*,\d*)', div_price[0]).groups()
		# substitute comma by dot to correctly parse float
		prices = tuple(map(lambda price: float(price.replace(',','.')), prices))
		if len(div_price) > 1:
			price_quantity = re.search(r'\(per (\d*.?)\)', div_price[1]).group(1)
		else:
			price_quantity = "plate"
		# allergens
		allergens = meal.xpath('./div[3]/div[1]/div[contains(@class, "azn") and\
			not(contains(@class, "ptr")) and\
				not(contains(./preceding-sibling::div/@class, "hidden"))]/text()')
		if len(allergens) > 8:
			# allergens are included
			allergens = [Allergen(allergen) for allergen in allergens[0].strip().strip('()')\
				.split(', ') if allergen != '' and allergen not in map(str, list(range(1,12)))]
		else:
			# there are no allergens
			allergens = None
		# some tags / binary slots
		tags = meal.xpath('./div[4]/div[2]/img/@title')
		vegan = 'vegan' in tags
		vegetarian = 'vegetarian' in tags
		fish = 'MSC (MSC-C-51632)' in tags
		pork = 'pork' in tags or 'beef/pork' in tags
		dish_type = meal.xpath('./preceding-sibling::div[contains(@class, "gruppenkopf")][1]\
			/div[contains(@class, "gruppenname")]/text()')[0]

		return Meal(name, day, prices, price_quantity, allergens, vegan, vegetarian, fish, pork,\
			DishType.from_website_name(dish_type))

	def _parse_date(self, date: str) -> datetime.datetime:
		"""Parse the given string as date. Allowed is a date given as Y-m-d or one of today,
		tomorrow and monday-sunday.
		
		Raises:
			ParseDateError: If ``date`` could not be parsed.
		"""

		try:
			# try to parse date
			date = datetime.datetime.strptime(date, '%Y-%m-%d')
		except ValueError:
			# cover some specific cases, e.g. today, tomorrow, wednesday
			weekdays =\
				['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']
			if date == 'today':
				date = datetime.datetime.today()
			elif date == 'tomorrow':
				date = datetime.datetime.today() + datetime.timedelta(days=1)
			elif date.lower() in weekdays:
				today = datetime.datetime.today().weekday()
				weekday = weekdays.index(date.lower())
				if weekday <= today:
					# if today is named shall we consider the weekday next week instead? (<= vs. <)
					weekday += 7
				date = datetime.datetime.today() + datetime.timedelta(days=weekday-today)
			else:
				raise ParseDateError

		return date

	def get_meals(self, date: str, use_cache: bool = True) -> List[Meal]:
		"""
		Gets the meals for a specified day by either looking them up in the cache or by issuing and
		parsing a post request.

		Args:
            date (str): The date for which the data will be returned.
				Can be a string in the format 'Y-m-d' or one of today, tomorrow and monday-sunday.
            use_cache (bool): If False will always query the server instead of using the cache.

        Returns:
            :obj:`list` of Meal: List of meals for specified date
		"""

		date = self._parse_date(date)
		if use_cache and date.date() in self.storage:
			# NOTE data could be deprecated
			return self.storage[date.date()]
		else:
			# issue request to server
			return self._parse(date)
