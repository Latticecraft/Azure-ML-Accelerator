import numpy as np
from woodwork.column_schema import ColumnSchema
from woodwork.logical_types import IntegerNullable

from featuretools.primitives.base import TransformPrimitive


class PercentChange(TransformPrimitive):
    """Calculates the cumulative sum.
    Description:
        Given a list of values, return the cumulative sum
        (or running total). There is no set window, so the
        sum at each point is calculated over all prior values.
        `NaN` values will return `NaN`, but in the window of a
        cumulative caluclation, they're ignored.
    Examples:
        >>> cum_sum = CumSum()
        >>> cum_sum([1, 2, 3, 4, None, 5]).tolist()
        [1.0, 3.0, 6.0, 10.0, nan, 15.0]
    """

    name = "percent_change"
    input_types = [ColumnSchema(semantic_tags={"numeric"})]
    return_type = ColumnSchema(semantic_tags={"numeric"})
    uses_full_dataframe = False
    description_template = "the percent change of {}"

    def get_function(self):
        def percent_change(values):
            return values.cumsum()

        return percent_change