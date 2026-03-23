
from django.urls import path
from . import views

urlpatterns = [
    path("", views.index, name="index"),         # homepage
    path("chat/", views.chat_api, name="chat_api")  # AJAX API
]