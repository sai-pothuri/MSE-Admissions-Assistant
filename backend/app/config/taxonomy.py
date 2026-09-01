from enum import StrEnum


class Category(StrEnum):
    ADMISSIONS = "admissions"
    CURRICULUM = "curriculum"
    TUITION = "tuition"
    DEADLINES = "deadlines"
    FACULTY = "faculty"
    OTHER = "other"


CATEGORIES: list[str] = [c.value for c in Category]
