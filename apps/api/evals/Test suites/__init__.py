# Makes 'Test suites' importable as a Python sub-package.
# The directory name contains a space so direct dot-import is not valid;
# evals/runner.py uses importlib.util to load tests.py by file path instead.
