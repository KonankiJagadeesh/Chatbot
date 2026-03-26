
from django.urls import path
from . import views

urlpatterns = [
    path("", views.index, name="index"),
    path("chat/", views.chat_api, name="chat_api"),
    path("clear/", views.clear_session, name="clear_session"),
]