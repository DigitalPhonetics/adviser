from enum import Enum

class MessageCode(Enum):
    HELLO = 1
    WELCOME = 2
    ABORT = 3
    GOODBYE = 6
    ERROR = 8
    PUBLISH = 16
    PUBLISHED = 17
    SUBSCRIBE = 32
    SUBSCRIBED = 33
    UNSUBSCRIBE = 34
    UNSUBSCRIBED = 35
    EVENT = 36
    CALL = 48
    RESULT = 50
    REGISTER = 64
    REGISTERED = 65
    UNREGISTER = 66
    UNREGISTERED = 67
    INVOCATION = 68
    YIELD = 70