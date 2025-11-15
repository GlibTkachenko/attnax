# Contributing to Attnax

Thank you for your interest in contributing to Attnax!

## Code Contributions

All submissions require review. We use GitHub pull requests for this purpose.

1. Fork the repository
2. Create a branch for your changes
3. Install in development mode: `pip install -e .`
4. Make your changes
5. Run the tests locally
6. Submit a pull request

## Testing

Install the package in development mode first:

```bash
pip install -e .
```

Then run the tests:

```bash
python -m pytest tests/
```

Alternatively, you can run tests without pytest:

```bash
python -m tests.test_components
python -m tests.test_training
```

### For Your Own Projects

If you want to use Attnax as a base for your own project, feel free to fork the repository. The Apache 2.0 license allows you to modify and use the code freely.
