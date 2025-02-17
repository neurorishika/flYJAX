# simulation/parse.py
import re
import jax.numpy as jnp

def parse_reward_matrix(reward_str: str) -> jnp.ndarray:
    """
    Parse a reward string into a reward matrix.
    """
    segments = reward_str.split(';')
    rows = []
    for seg in segments:
        seg = seg.strip()
        match = re.match(r"\[(.*?)\]x(\d+)", seg)
        if not match:
            raise ValueError(f"Segment '{seg}' is not in the expected format '[a,b]xn'")
        numbers_str = match.group(1)
        repeat_count = int(match.group(2))
        try:
            numbers = [float(num.strip()) for num in numbers_str.split(',')]
        except ValueError:
            raise ValueError(f"Could not convert numbers in segment '{seg}' to floats.")
        if len(numbers) != 2:
            raise ValueError(f"Segment '{seg}' must contain exactly two numbers.")
        for _ in range(repeat_count):
            rows.append(numbers)
    return jnp.array(rows)
