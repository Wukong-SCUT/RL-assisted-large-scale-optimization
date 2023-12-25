(function(){
  window.rollout = {
    {% for feature, is_active in features.items %}
      "{{ feature }}": {{ is_active|yesno:"true,false" }},
    {% endfor %}
  };
})();