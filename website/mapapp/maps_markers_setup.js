window.onload = function initMap() {
  var uluru = {lat: 51.529517, lng: -0.058284};
  var styledMapType = new google.maps.StyledMapType(
    [
  {
    "elementType": "geometry",
    "stylers": [
      {
        "color": "#f5f5f5"
      }
    ]
  },
  {
    "elementType": "labels.icon",
    "stylers": [
      {
        "visibility": "off"
      }
    ]
  },
  {
    "elementType": "labels.text.fill",
    "stylers": [
      {
        "color": "#616161"
      }
    ]
  },
  {
    "elementType": "labels.text.stroke",
    "stylers": [
      {
        "color": "#f5f5f5"
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
    "elementType": "labels.text.fill",
    "stylers": [
      {
        "color": "#bdbdbd"
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
    "featureType": "poi",
    "elementType": "geometry",
    "stylers": [
      {
        "color": "#eeeeee"
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
        "color": "#757575"
      }
    ]
  },
  {
    "featureType": "poi.park",
    "elementType": "geometry",
    "stylers": [
      {
        "color": "#e5e5e5"
      }
    ]
  },
  {
    "featureType": "poi.park",
    "elementType": "labels.text.fill",
    "stylers": [
      {
        "color": "#9e9e9e"
      }
    ]
  },
  {
    "featureType": "road",
    "elementType": "geometry",
    "stylers": [
      {
        "color": "#ffffff"
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
    "featureType": "road.arterial",
    "stylers": [
      {
        "visibility": "off"
      }
    ]
  },
  {
    "featureType": "road.arterial",
    "elementType": "labels.text.fill",
    "stylers": [
      {
        "color": "#757575"
      }
    ]
  },
  {
    "featureType": "road.highway",
    "elementType": "geometry",
    "stylers": [
      {
        "color": "#dadada"
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
    "featureType": "road.highway",
    "elementType": "labels.text.fill",
    "stylers": [
      {
        "color": "#616161"
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
        "color": "#9e9e9e"
      }
    ]
  },
  {
    "featureType": "transit.line",
    "elementType": "geometry",
    "stylers": [
      {
        "color": "#e5e5e5"
      }
    ]
  },
  {
    "featureType": "transit.station",
    "elementType": "geometry",
    "stylers": [
      {
        "color": "#eeeeee"
      }
    ]
  },
  {
    "featureType": "water",
    "elementType": "geometry",
    "stylers": [
      {
        "color": "#c9c9c9"
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
        "color": "#9e9e9e"
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
          url: './assets/charts/' + i + '.png',
          scaledSize: new google.maps.Size(25, 25), // scaled size
      },
    });
    //
    // google.maps.event.addListener(marker, 'mouseover', function() {
    //   this.setIcon({
    //     path: google.maps.SymbolPath.CIRCLE,
    //     scale: 8
    //   });
    // });
    // google.maps.event.addListener(marker, 'mouseout', function() {
    //   this.setIcon({
    //     path: google.maps.SymbolPath.CIRCLE,
    //     scale: 5
    //   });
    // });
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
