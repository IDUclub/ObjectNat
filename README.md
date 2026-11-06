# Repository Coverage

[Full report](https://htmlpreview.github.io/?https://github.com/IDUclub/ObjectNat/blob/python-coverage-comment-action-data/htmlcov/index.html)

| Name                                                     |    Stmts |     Miss |   Branch |   BrPart |   Cover |   Missing |
|--------------------------------------------------------- | -------: | -------: | -------: | -------: | ------: | --------: |
| objectnat/\_\_init\_\_.py                                |        3 |        0 |        0 |        0 |    100% |           |
| objectnat/\_api.py                                       |        6 |        0 |        0 |        0 |    100% |           |
| objectnat/\_config.py                                    |       14 |        0 |        0 |        0 |    100% |           |
| objectnat/\_version.py                                   |        1 |        0 |        0 |        0 |    100% |           |
| objectnat/methods/\_\_init\_\_.py                        |        0 |        0 |        0 |        0 |    100% |           |
| objectnat/methods/accessibility/\_\_init\_\_.py          |        3 |        0 |        0 |        0 |    100% |           |
| objectnat/methods/accessibility/\_utils.py               |      168 |       28 |       74 |       31 |     76% |20, 22, 24, 53, 55, 57, 75, 106, 120-\>122, 161, 179-\>183, 181, 219, 227, 231, 235, 256, 280, 283, 285, 287, 289, 291, 295, 300-\>304, 311, 316, 335, 350, 376, 379 |
| objectnat/methods/accessibility/coverage.py              |       44 |        4 |       14 |        4 |     86% |123, 125, 152, 180 |
| objectnat/methods/accessibility/isochrones.py            |       64 |        5 |       22 |        6 |     87% |173, 175, 203, 209-\>220, 223, 249 |
| objectnat/methods/accessibility/radius.py                |       14 |        0 |        0 |        0 |    100% |           |
| objectnat/methods/insolation/\_\_init\_\_.py             |        1 |        0 |        0 |        0 |    100% |           |
| objectnat/methods/insolation/\_sun.py                    |       43 |        1 |        2 |        1 |     96% |        83 |
| objectnat/methods/insolation/insolation.py               |      141 |        8 |       36 |       12 |     89% |134, 246-\>249, 250, 254-\>260, 261, 272, 275, 307-\>309, 380, 383, 385, 387-\>389 |
| objectnat/methods/noise/\_\_init\_\_.py                  |        4 |        0 |        0 |        0 |    100% |           |
| objectnat/methods/noise/noise\_grid.py                   |      210 |       32 |       58 |       21 |     79% |31-36, 149, 151, 153, 155, 165, 168, 227-228, 230-233, 248-249, 280, 301-\>325, 311-\>322, 316-\>322, 326-331, 333-\>340, 347, 351, 385, 399, 427 |
| objectnat/methods/noise/noise\_init\_data.py             |        3 |        0 |        0 |        0 |    100% |           |
| objectnat/methods/noise/noise\_reduce.py                 |       47 |        2 |       20 |        2 |     94% |  125, 128 |
| objectnat/methods/noise/noise\_simulation.py             |      364 |       23 |      122 |       20 |     91% |144, 208, 253, 260, 264, 272, 274, 279, 284, 286, 288, 297, 302, 313, 356-357, 401, 462-463, 470-\>465, 473, 525, 530, 549 |
| objectnat/methods/noise/noise\_simulation\_simplified.py |       94 |        3 |       36 |        3 |     95% |192, 197, 232 |
| objectnat/methods/provision/\_\_init\_\_.py              |        2 |        0 |        0 |        0 |    100% |           |
| objectnat/methods/provision/\_matrix.py                  |       40 |        1 |       10 |        1 |     96% |        40 |
| objectnat/methods/provision/provision.py                 |       20 |        0 |        2 |        0 |    100% |           |
| objectnat/methods/provision/provision\_calculating.py    |      122 |        5 |       44 |        5 |     94% |27, 29, 43, 45, 194 |
| objectnat/methods/provision/provision\_exceptions.py     |       36 |        0 |       16 |        0 |    100% |           |
| objectnat/methods/provision/provision\_result.py         |       77 |        8 |       38 |        8 |     86% |56, 58, 60, 112, 134, 138, 140, 172 |
| objectnat/methods/utils/\_\_init\_\_.py                  |        0 |        0 |        0 |        0 |    100% |           |
| objectnat/methods/utils/geom\_utils.py                   |       94 |        8 |       22 |        5 |     89% |21, 62, 69-71, 81-82, 98 |
| objectnat/methods/utils/raster.py                        |      177 |       90 |       44 |       10 |     49% |19, 47, 58, 71-\>68, 113-115, 120-121, 133-140, 144-163, 189-203, 242-262, 266-289, 321, 323, 329-336, 338, 341, 343 |
| objectnat/methods/utils/scene.py                         |      162 |       29 |       62 |       22 |     77% |45, 54, 85-\>87, 92, 99, 101, 105-\>107, 108-109, 112, 115, 121, 123, 130, 133-139, 146, 148, 154, 159-160, 176, 178, 185, 187-188 |
| objectnat/methods/visibility/\_\_init\_\_.py             |        3 |        0 |        0 |        0 |    100% |           |
| objectnat/methods/visibility/\_native.py                 |       83 |        2 |       24 |        3 |     95% |112-\>114, 132, 150 |
| objectnat/methods/visibility/catchment.py                |      101 |       14 |       36 |       10 |     81% |73, 75, 77, 85, 88, 102, 112, 120, 163, 167, 209-212 |
| objectnat/methods/visibility/isovist.py                  |       89 |        2 |        4 |        1 |     97% |     72-73 |
| objectnat/methods/visibility/viewshed.py                 |      131 |       20 |       42 |       12 |     78% |122-\>119, 205-210, 215-\>220, 218, 221, 223, 253, 264, 266, 270, 285-290, 303 |
| objectnat/methods/visibility/visibility\_analysis.py     |      100 |       10 |       38 |       11 |     85% |144, 147, 166, 232, 234, 237, 240, 243, 247, 256-\>263, 265 |
| **TOTAL**                                                | **2461** |  **295** |  **766** |  **188** | **84%** |           |


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