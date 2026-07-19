__all__ = ("create_env",)


def __getattr__(name: str):
    if name == "create_env":
        from .create_env import create_env

        return create_env
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
