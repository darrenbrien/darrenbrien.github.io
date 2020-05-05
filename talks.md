---
layout: default
title: Talks
---
# Talks

## Upcoming
<ul>
  {% for talk in site.data.upcoming %}
    <li>
      <h2>{{ talk.title }} - {{ talk.date }}</h2>
      {{ talk.excerpt }}
    </li>
  {% endfor %}
</ul>

## Past
<ul>
  {% for talk in site.data.talks %}
    <li>
      <h2>{{ talk.title }} - {{ talk.date }}</h2>
      {{ talk.excerpt }}
    </li>
  {% endfor %}
</ul>
