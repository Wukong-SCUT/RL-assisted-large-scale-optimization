# -*- coding: utf-8 -*-
from django.utils.decorators import method_decorator
from django.views.decorators.cache import never_cache
from django.views.generic.base import TemplateView

from . import rollout as proclaim


class RolloutView(TemplateView):
    template_name = "rollout/rollout.js"

    @method_decorator(never_cache)
    def dispatch(self, *args, **kwargs):
        return super(RolloutView, self).dispatch(*args, **kwargs)

    def render_to_response(self, context, **kwargs):
        kwargs['content_type'] = 'application/x-javascript'
        return super(RolloutView, self).render_to_response(context, **kwargs)

    def get_context_data(self, **kwargs):
        kwargs['features'] = dict([(feature, proclaim.is_active(feature, self.request.user))
            for feature in proclaim.features()])
        return kwargs
