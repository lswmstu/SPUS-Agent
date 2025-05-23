"""
URL configuration for predictCoze project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path

from DjangoWeb import views

urlpatterns = [
    path('admin/', admin.site.urls),
    path('predict/', views.predict),
    path('draw_star_chart/', views.draw_star_chart),
    path('panoramicMapWalking/', views.panoramicMapWalking),
    path('panoramicMapWalkingMove/', views.panoramicMapWalkingMove),
    path('predict_move',views.predict_move),
    path('get_location_map',views.get_location_map)

]
