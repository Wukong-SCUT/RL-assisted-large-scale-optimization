# -*- coding: utf-8 -*-
from django.conf.urls.defaults import *

from rollout.views import RolloutView


urlpatterns = patterns('',
    url(r'^rollout\.js$', RolloutView.as_view(), name="rollout_js"),
)
