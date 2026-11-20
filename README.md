# Repository Coverage

[Full report](https://htmlpreview.github.io/?https://github.com/IDUclub/ObjectNat/blob/python-coverage-comment-action-data/htmlcov/index.html)

| Name                                                     |    Stmts |     Miss |   Branch |   BrPart |   Cover |   Missing |
|--------------------------------------------------------- | -------: | -------: | -------: | -------: | ------: | --------: |
| objectnat/\_\_init\_\_.py                                |        3 |        0 |        0 |        0 |    100% |           |
| objectnat/\_api.py                                       |        7 |        0 |        0 |        0 |    100% |           |
| objectnat/\_config.py                                    |       14 |        0 |        0 |        0 |    100% |           |
| objectnat/\_version.py                                   |        1 |        0 |        0 |        0 |    100% |           |
| objectnat/methods/\_\_init\_\_.py                        |        0 |        0 |        0 |        0 |    100% |           |
| objectnat/methods/accessibility/\_\_init\_\_.py          |        3 |        0 |        0 |        0 |    100% |           |
| objectnat/methods/accessibility/\_geometry.py            |      132 |        9 |       28 |        7 |     90% |42, 46, 107, 122, 134, 170, 176-178 |
| objectnat/methods/accessibility/\_utils.py               |      223 |       37 |       80 |       31 |     76% |22, 24, 26, 55, 57, 59, 77, 114, 128-\>130, 137, 143, 147, 151, 154-155, 158, 164, 198-200, 213-216, 228-\>230, 246-248, 265-\>269, 267, 347, 349, 351, 353, 355, 362-\>366, 385, 405, 422, 448, 451 |
| objectnat/methods/accessibility/coverage.py              |       44 |        4 |       14 |        4 |     86% |124, 126, 153, 184 |
| objectnat/methods/accessibility/isochrones.py            |       66 |        6 |       20 |        6 |     86% |171, 173, 202, 209-\>220, 225-233, 244 |
| objectnat/methods/accessibility/radius.py                |       14 |        0 |        0 |        0 |    100% |           |
| objectnat/methods/insolation/\_\_init\_\_.py             |        1 |        0 |        0 |        0 |    100% |           |
| objectnat/methods/insolation/\_points.py                 |       75 |        3 |       28 |        4 |     93% |28, 37, 67, 94-\>100 |
| objectnat/methods/insolation/\_sun.py                    |       43 |        1 |        2 |        1 |     96% |        83 |
| objectnat/methods/insolation/\_trees.py                  |      134 |        2 |       40 |        0 |     99% |     60-61 |
| objectnat/methods/insolation/insolation.py               |      219 |        8 |       62 |       11 |     93% |204-\>210, 208-\>210, 224, 418, 429, 479-\>485, 486, 500, 644, 647, 649 |
| objectnat/methods/noise/\_\_init\_\_.py                  |        4 |        0 |        0 |        0 |    100% |           |
| objectnat/methods/noise/noise\_grid.py                   |      210 |       32 |       58 |       21 |     79% |31-36, 149, 151, 153, 155, 165, 168, 227-228, 230-233, 248-249, 280, 301-\>325, 311-\>322, 316-\>322, 326-331, 333-\>340, 347, 351, 385, 399, 427 |
| objectnat/methods/noise/noise\_init\_data.py             |        3 |        0 |        0 |        0 |    100% |           |
| objectnat/methods/noise/noise\_reduce.py                 |       47 |        2 |       20 |        2 |     94% |  125, 128 |
| objectnat/methods/noise/noise\_simulation.py             |      364 |       23 |      122 |       20 |     91% |144, 208, 253, 260, 264, 272, 274, 279, 284, 286, 288, 297, 302, 313, 356-357, 401, 462-463, 470-\>465, 473, 525, 530, 549 |
| objectnat/methods/noise/noise\_simulation\_simplified.py |       94 |        3 |       36 |        3 |     95% |187, 192, 225 |
| objectnat/methods/provision/\_\_init\_\_.py              |        2 |        0 |        0 |        0 |    100% |           |
| objectnat/methods/provision/\_matrix.py                  |       40 |        1 |       10 |        1 |     96% |        40 |
| objectnat/methods/provision/provision.py                 |       20 |        0 |        2 |        0 |    100% |           |
| objectnat/methods/provision/provision\_calculating.py    |      122 |        5 |       44 |        5 |     94% |27, 29, 43, 45, 194 |
| objectnat/methods/provision/provision\_exceptions.py     |       36 |        0 |       16 |        0 |    100% |           |
| objectnat/methods/provision/provision\_result.py         |       77 |        8 |       38 |        8 |     86% |56, 58, 60, 112, 134, 138, 140, 172 |
| objectnat/methods/utils/\_\_init\_\_.py                  |        0 |        0 |        0 |        0 |    100% |           |
| objectnat/methods/utils/\_building\_details.py           |      261 |        3 |       46 |        4 |     98% |204, 235-\>238, 336, 338 |
| objectnat/methods/utils/geom\_utils.py                   |       94 |        8 |       22 |        5 |     89% |21, 62, 69-71, 81-82, 98 |
| objectnat/methods/utils/raster.py                        |      177 |       90 |       44 |       10 |     49% |19, 47, 58, 71-\>68, 113-115, 120-121, 133-140, 144-163, 189-203, 242-262, 266-289, 321, 323, 329-336, 338, 341, 343 |
| objectnat/methods/utils/scene.py                         |      249 |       26 |       92 |       19 |     87% |133, 142, 178, 184-\>186, 256, 263, 265, 269-\>271, 272-273, 279, 294, 297-303, 310, 312, 318, 323-324, 340, 342, 351-352 |
| objectnat/methods/utils/skeleton.py                      |       60 |       50 |       20 |        0 |     12% |    83-152 |
| objectnat/methods/visibility/\_\_init\_\_.py             |        3 |        0 |        0 |        0 |    100% |           |
| objectnat/methods/visibility/\_native.py                 |       83 |        2 |       24 |        3 |     95% |112-\>114, 132, 150 |
| objectnat/methods/visibility/catchment.py                |      101 |       14 |       36 |       10 |     81% |73, 75, 77, 85, 88, 102, 112, 120, 163, 167, 209-212 |
| objectnat/methods/visibility/isovist.py                  |       89 |        2 |        4 |        1 |     97% |     72-73 |
| objectnat/methods/visibility/viewshed.py                 |      131 |       20 |       42 |       12 |     78% |122-\>119, 205-210, 215-\>220, 218, 221, 223, 253, 264, 266, 270, 285-290, 303 |
| objectnat/methods/visibility/visibility\_analysis.py     |       93 |       10 |       34 |       11 |     83% |130, 133, 143, 209, 211, 214, 217, 220, 224, 233-\>240, 242 |
| **TOTAL**                                                | **3339** |  **369** |  **984** |  **199** | **86%** |           |


## Setup coverage badge

Below are examples of the badges you can use in your main branch `README` file.

### Direct image

[![Coverage badge](https://raw.githubusercontent.com/IDUclub/ObjectNat/python-coverage-comment-action-data/badge.svg)](https://htmlpreview.github.io/?https://github.com/IDUclub/ObjectNat/blob/python-coverage-comment-action-data/htmlcov/index.html)

This is the one to use if your repository is private or if you don't want to customize anything.

### [Shields.io](https://shields.io) Json Endpoint

[![Coverage badge](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/IDUclub/ObjectNat/python-coverage-comment-action-data/endpoint.json)](https://htmlpreview.github.io/?https://github.com/IDUclub/ObjectNat/blob/python-coverage-comment-action-data/htmlcov/index.html)

Using this one will allow you to [customize](https://shields.io/endpoint) the look of your badge.
It won't work with private repositories. It won't be refreshed more than once per five minutes.

### [Shields.io](https://shields.io) Dynamic Badge

[![Coverage badge](https://img.shields.io/badge/dynamic/json?color=brightgreen&label=coverage&query=%24.message&url=https%3A%2F%2Fraw.githubusercontent.com%2FIDUclub%2FObjectNat%2Fpython-coverage-comment-action-data%2Fendpoint.json)](https://htmlpreview.github.io/?https://github.com/IDUclub/ObjectNat/blob/python-coverage-comment-action-data/htmlcov/index.html)

This one will always be the same color. It won't work for private repos. I'm not even sure why we included it.

## What is that?

This branch is part of the
[python-coverage-comment-action](https://github.com/marketplace/actions/python-coverage-comment)
GitHub Action. All the files in this branch are automatically generated and may be
overwritten at any moment.