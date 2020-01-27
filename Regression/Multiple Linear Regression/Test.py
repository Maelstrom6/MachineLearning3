import math

def approximate_pi(n):
    # We have n by n pixels on the first quadrant
    # X is the number of points that lie in the circle
    x = 0
    for x_pixel in range(n):
        for y_pixel in range(n):
            x_coord = x_pixel/n + 1/(2*n)
            y_coord = y_pixel/n + 1/(2*n)
            # These are now the coordinates of the centre of each pixel
            if x_coord**2+y_coord**2 < 1:
                x += 1
    return 4*x/n**2


def squared_error(n):
    return (approximate_pi(n)-math.pi)**2


N = 1000000
print(squared_error(int(N**0.5)))
print(math.pi*(4-math.pi)/N)