import pandas as pd
import pm4py

from pm4py.objects.log.importer.xes import factory as xes_import_factory
log = xes_import_factory.apply("BPI Challenge 2018.xes")

#Test 1