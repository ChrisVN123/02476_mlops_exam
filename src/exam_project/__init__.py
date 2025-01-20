import os

_EXAM_FOLDER_ROOT = os.path.dirname(__file__)  # root of test folder
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_EXAM_FOLDER_ROOT))  # root of project
_PATH_DATA = os.path.join(_PROJECT_ROOT, r"data/")  # root of data
print(
    "\nProject root",
    _PROJECT_ROOT,
    "\nPath data",
    _PATH_DATA,
    "\nExam root",
    _EXAM_FOLDER_ROOT,
)
