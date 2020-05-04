# FAQ

## Who contributed to ADvISER?
!!! todo
    Add missing members

- Daniel Ortega
- Dirk Väth
- Gianna Weber
- Lindsey Vanderlyn
- Maximilian Schmidt
- Moritz Völkel
- Zorica Kacarevic
- Ngoc Thang Vu

## How shall I cite ADvISER

!!! todo
    Add links

Please see here.
Who can I contact in case of problems or questions?

You can ask questions by sending emails to adviser-support@ims.uni-stuttgart.de

You can also post bug reports and feature requests (only) in GitHub issues. Make sure to read our guidelines first.


## Can I contribute to the project?

!!! todo
    Add links

You can post bug reports and feature requests in GitHub issues. You can find the code to ADvISER in our Git repository.

Information about the download can be found here.


<!-- ##System Specific Information -->

## What are the main features of the system’s framework?
!!! todo
    Update/Add image

_images/sds.png

## Which User Actions and System Actions are currently supported by the system?

### User Actions
* **Inform**: User informs the system about a constraint/entity name
* **NegativeInform**: User informs the system they do not want a particular value
* **Request**: User asks the system for information about an entity
* **Hello**: User issues a greeting
* **Bye**: User says bye; this ends the dialog
* **Thanks**: User says thanks
* **Affirm**: User agrees with the last system confirm request
* **Deny**: User disagrees with the last system confirm request
* **RequestAlternatives**: User asks for an alternative offer from the system
* **Ack**: User likes the system's proposed offer
* **Bad**: User input could not be recognized
* **SelectDomain**: User has provided a domain keyword

### System Actions
* **Welcome**: Issue system greeting
* **InformByName**: Propose an entity to the user
*  **InformByAlternatives**: Propose an alternate entity if the user isn't satisfied with the first
* **Request**: Ask for more information from the user
* **Confirm**: Ask the user to confirm a proposed value for a slot
* **Select**: Provide the user with 2 or 3 options and ask the user to select the correct one
* **RequestMore**: Ask the user if there is anything else the system can provide
* **Bad**: If the system could not understand the user
* **Bye**: Say goodbye

## What Emotions and Engagements are currently supported by the system?

### User Emotions
* **happy**
* **angry**
* **neutral**

### User Engagement
* **high**
* **low**

## Which domains are currently supported by ADvISER?
!!! todo
    Update domains

ADvISER currently supports the following domains:

IMS Lecturers

    Providing information about lecturers teaching at the IMS (for privacy reasons, our database includes fictive information about lecturers and related contact information, however, it serves as an example for a real-world application).

IMS Courses

    Providing information about courses offered at the IMS, e.g. course content, requirements, or locational information.

## Can ADvISER be extended by new modules?
!!! todo
    Add link

Please see here.
