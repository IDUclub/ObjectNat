import os
import sys
from datetime import datetime
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as pkg_version

from pygments.lexers import get_lexer_by_name
from sphinx.highlighting import lexers

sys.path.insert(0, os.path.abspath(".."))

project = "ObjectNat"
author = "Donny"
copyright = f"{datetime.now():%Y}, {author}"

try:
    release = pkg_version("objectnat")
except PackageNotFoundError:
    release = "0.0.0"

version = ".".join(release.split(".")[:2])


extensions = [
    "myst_nb",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx_copybutton",
    "sphinx_autodoc_typehints",
    "sphinx_design",
]

html_theme = "furo"
html_static_path = ["_static"]
html_css_files = ["lang.css"]
templates_path = ["_templates"]
# the language switch sits under the logo; the other entries are furo's defaults
html_sidebars = {
    "**": [
        "sidebar/brand.html",
        "sidebar/lang.html",
        "sidebar/search.html",
        "sidebar/scroll-start.html",
        "sidebar/navigation.html",
        "sidebar/ethical-ads.html",
        "sidebar/scroll-end.html",
        "sidebar/variant-selector.html",
    ]
}

# Two languages: English sources, Russian translations in locale/ru/LC_MESSAGES/<page>.po (one catalogue per
# page). Build the Russian site with ``-D language=ru`` into html/ru; ``python docs/i18n.py update`` adds new or
# changed text to the catalogues and ``python docs/i18n.py check`` fails while anything is left untranslated.
language = "en"
locale_dirs = ["locale/"]
gettext_compact = False
gettext_location = False
gettext_uuid = False

html_show_sourcelink = True

html_logo = "_static/ONlogo.svg"
html_favicon = "_static/ONFavicon.svg"

myst_enable_extensions = ["colon_fence", "deflist", "substitution", "linkify", "attrs_block", "attrs_inline"]
nb_execution_mode = "off"
nb_execution_timeout = 600
nb_execution_raise_on_error = False
nb_render_image_options = {"align": "center"}

autosummary_generate = True
autodoc_typehints = "description"
autodoc_member_order = "bysource"


intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/stable/", None),
    "geopandas": ("https://geopandas.org/en/stable/", None),
    "shapely": ("https://shapely.readthedocs.io/en/stable/", None),
}


napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_preprocess_types = True

napoleon_type_aliases = {
    "gpd.GeoDataFrame": "geopandas.GeoDataFrame",
    "GeoDataFrame": "geopandas.GeoDataFrame",
    "Series": "pandas.Series",
    "DataFrame": "pandas.DataFrame",
    "LineString": "shapely.geometry.LineString",
}

exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    "methods/clustering.rst",
    "methods/examples/clustering.ipynb",
]
lexers["ipython2"] = get_lexer_by_name("ipython3")
