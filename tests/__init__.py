import os

_TEST_ROOT = os.path.dirname(__file__)  # root of test folder
_PROJECT_ROOT = os.path.dirname(_TEST_ROOT)  # root of project
_PATH_DATA = os.path.join(_PROJECT_ROOT, r"data")  # root of data
print(
    "\nProject root",
    _PROJECT_ROOT,
    "\nPath data",
    _PATH_DATA,
    "\nTest root",
    _TEST_ROOT,
)
