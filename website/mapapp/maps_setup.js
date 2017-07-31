window.onload = function initMap() {
  var uluru = {lat: 51.529517, lng: -0.058284};
  console.log("sds")
  console.log(JSON.parse('style.json'))
  var styledMapType = new google.maps.StyledMapType(
    [
      {
        "elementType": "geometry",
        "stylers": [
          {
            "color": "#ebe3cd"
          }
        ]
      },
      {
        "elementType": "labels.text.fill",
        "stylers": [
          {
            "color": "#523735"
          }
        ]
      },
      {
        "elementType": "labels.text.stroke",
        "stylers": [
          {
            "color": "#f5f1e6"
          }
        ]
      },
      {
        "featureType": "administrative",
        "elementType": "geometry.stroke",
        "stylers": [
          {
            "color": "#c9b2a6"
          }
        ]
      },
      {
        "featureType": "administrative.land_parcel",
        "stylers": [
          {
            "visibility": "off"
          }
        ]
      },
      {
        "featureType": "administrative.land_parcel",
        "elementType": "geometry.stroke",
        "stylers": [
          {
            "color": "#dcd2be"
          }
        ]
      },
      {
        "featureType": "administrative.land_parcel",
        "elementType": "labels.text.fill",
        "stylers": [
          {
            "color": "#ae9e90"
          }
        ]
      },
      {
        "featureType": "administrative.neighborhood",
        "stylers": [
          {
            "visibility": "off"
          }
        ]
      },
      {
        "featureType": "landscape.natural",
        "elementType": "geometry",
        "stylers": [
          {
            "color": "#dfd2ae"
          }
        ]
      },
      {
        "featureType": "poi",
        "elementType": "geometry",
        "stylers": [
          {
            "color": "#dfd2ae"
          }
        ]
      },
      {
        "featureType": "poi",
        "elementType": "labels.text",
        "stylers": [
          {
            "visibility": "off"
          }
        ]
      },
      {
        "featureType": "poi",
        "elementType": "labels.text.fill",
        "stylers": [
          {
            "color": "#93817c"
          }
        ]
      },
      {
        "featureType": "poi.business",
        "stylers": [
          {
            "visibility": "off"
          }
        ]
      },
      {
        "featureType": "poi.park",
        "elementType": "geometry.fill",
        "stylers": [
          {
            "color": "#a5b076"
          }
        ]
      },
      {
        "featureType": "poi.park",
        "elementType": "labels.text.fill",
        "stylers": [
          {
            "color": "#447530"
          }
        ]
      },
      {
        "featureType": "road",
        "elementType": "geometry",
        "stylers": [
          {
            "color": "#f5f1e6"
          }
        ]
      },
      {
        "featureType": "road",
        "elementType": "labels",
        "stylers": [
          {
            "visibility": "off"
          }
        ]
      },
      {
        "featureType": "road",
        "elementType": "labels.icon",
        "stylers": [
          {
            "visibility": "off"
          }
        ]
      },
      {
        "featureType": "road.arterial",
        "stylers": [
          {
            "visibility": "off"
          }
        ]
      },
      {
        "featureType": "road.arterial",
        "elementType": "geometry",
        "stylers": [
          {
            "color": "#fdfcf8"
          }
        ]
      },
      {
        "featureType": "road.highway",
        "elementType": "geometry",
        "stylers": [
          {
            "color": "#f8c967"
          }
        ]
      },
      {
        "featureType": "road.highway",
        "elementType": "geometry.stroke",
        "stylers": [
          {
            "color": "#e9bc62"
          }
        ]
      },
      {
        "featureType": "road.highway",
        "elementType": "labels",
        "stylers": [
          {
            "visibility": "off"
          }
        ]
      },
      {
        "featureType": "road.highway.controlled_access",
        "elementType": "geometry",
        "stylers": [
          {
            "color": "#e98d58"
          }
        ]
      },
      {
        "featureType": "road.highway.controlled_access",
        "elementType": "geometry.stroke",
        "stylers": [
          {
            "color": "#db8555"
          }
        ]
      },
      {
        "featureType": "road.local",
        "stylers": [
          {
            "visibility": "off"
          }
        ]
      },
      {
        "featureType": "road.local",
        "elementType": "labels.text.fill",
        "stylers": [
          {
            "color": "#806b63"
          }
        ]
      },
      {
        "featureType": "transit",
        "stylers": [
          {
            "visibility": "off"
          }
        ]
      },
      {
        "featureType": "transit.line",
        "elementType": "geometry",
        "stylers": [
          {
            "color": "#dfd2ae"
          }
        ]
      },
      {
        "featureType": "transit.line",
        "elementType": "labels.text.fill",
        "stylers": [
          {
            "color": "#8f7d77"
          }
        ]
      },
      {
        "featureType": "transit.line",
        "elementType": "labels.text.stroke",
        "stylers": [
          {
            "color": "#ebe3cd"
          }
        ]
      },
      {
        "featureType": "transit.station",
        "elementType": "geometry",
        "stylers": [
          {
            "color": "#dfd2ae"
          }
        ]
      },
      {
        "featureType": "water",
        "elementType": "geometry.fill",
        "stylers": [
          {
            "color": "#b9d3c2"
          }
        ]
      },
      {
        "featureType": "water",
        "elementType": "labels.text",
        "stylers": [
          {
            "visibility": "off"
          }
        ]
      },
      {
        "featureType": "water",
        "elementType": "labels.text.fill",
        "stylers": [
          {
            "color": "#92998d"
          }
        ]
      }
    ]
    , {name: 'Map'});
  var map = new google.maps.Map(document.getElementById('map'), {
    zoom: 10,
    center: uluru,
    mapTypeControlOptions: {
      mapTypeIds: ['satellite', 'styled_map']
    }
  });
  var map_inset = new google.maps.Map(document.getElementById('map_inset'), {
    center: uluru,
    zoom: 18,
    mapTypeId: 'satellite',
    disableDefaultUI: true,
    scrollwheel: false,
    draggable: false
  });

  //Associate the styled map with the MapTypeId and set it to display.
  map.mapTypes.set('styled_map', styledMapType);
  map.setMapTypeId('styled_map');

  // Following blocks add the marker for the bethanl green site
  var marker = new google.maps.Marker({
    position: uluru,
    map: map,
    icon: {
      path: google.maps.SymbolPath.CIRCLE,
      scale: 5
    },
  });

  marker.addListener('click', function() {
    // map.setZoom(12);
    // map.setCenter(marker.getPosition());
    $('#myModal').modal('show');
  });

  google.maps.event.addListener(marker, 'mouseover', function() {
    this.setIcon({
      path: google.maps.SymbolPath.CIRCLE,
      scale: 8
    });
  });
  google.maps.event.addListener(marker, 'mouseout', function() {
    this.setIcon({
      path: google.maps.SymbolPath.CIRCLE,
      scale: 5
    });
  });

  // add a load of random markers
  for (i=0; i < 50; i++)
  {
    var marker = new google.maps.Marker({
      position: {lat: 51.277 + Math.random() * 0.35, lng: -0.4593 + Math.random() * 0.8},
      // 51.529517, lng: -0.058284};
      map: map,
      icon: {
        path: google.maps.SymbolPath.CIRCLE,
        scale: 5
      },
    });

    google.maps.event.addListener(marker, 'mouseover', function() {
      this.setIcon({
        path: google.maps.SymbolPath.CIRCLE,
        scale: 8
      });
    });
    google.maps.event.addListener(marker, 'mouseout', function() {
      this.setIcon({
        path: google.maps.SymbolPath.CIRCLE,
        scale: 5
      });
    });
    //
    // marker.addListener('click', function() {
    //   map.setZoom(12);
    //   map.setCenter(marker.getPosition());
    //   $('#myModal').modal('show');
    // });
  }

  // Resize map to show on a Bootstrap's modal
  $('#myModal').on('shown.bs.modal', function() {
    var currentCenter = map_inset.getCenter();  // Get current center before resizing
    google.maps.event.trigger(map_inset, "resize");
    // console.log(uluru)
    map_inset.setCenter(currentCenter); // Re-set previous center
  });



}
