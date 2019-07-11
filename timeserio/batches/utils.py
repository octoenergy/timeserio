def ceiling_division(dividend, divisor):
    if not dividend:
        return 0
    return dividend // divisor + (1 if dividend % divisor else 0)
