import numpy
import typing 

# metrics for measuring similarity between 2 image parts

def sum_of_squared_diffs(ints1, ints2):
    return numpy.sum(
        (ints1.flatten() - ints2.flatten()) ** 2
    )

def sum_of_absolute_diffs(ints1, ints2):
    return numpy.sum(
        numpy.abs(ints1.flatten() - ints2.flatten())
    )

# template matching algorithm 

def template_matching(
    img: numpy.ndarray, 
    template: numpy.ndarray,
    metric: typing.Callable
):
    kernel_size = template.shape[0]
    best_metric_value = float('inf')

    for col in range(kernel_size, img.shape[0], 1):
        for row in range(kernel_size, img.shape[1], 1):

            img_part = img[col:col+kernel_size, row:row+kernel_size]
            metric_value = metric(img_part, template)

            if metric_value < best_metric_value:
                found_template = img_part
                best_metric_value = metric_value

    return found_template, best_metric_value
