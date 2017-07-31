window.onload = function initMap() {

  function loadJSON(callback) {

    var xobj = new XMLHttpRequest();
    xobj.overrideMimeType("application/json");
    xobj.open('GET', 'style.json', true);
    xobj.onreadystatechange = function() {
      if (xobj.readyState == 4 && xobj.status == "200") {
        callback(xobj.responseText);
      }
    };
    xobj.send(null);
  }

  var london_centre = {lat: 51.529517, lng: -0.058284};

  // We create the map in the json loader callback, beacuse javascript is strange
  loadJSON(function(response) {
    loaded_json = JSON.parse(response);
    var styledMapType = new google.maps.StyledMapType(loaded_json, {
      name: 'Map'
    });

    var map = new google.maps.Map(document.getElementById('map'), {
      zoom: 10,
      center: london_centre,
      mapTypeControlOptions: {
        mapTypeIds: ['satellite', 'styled_map']
      }
    });

    //Associate the styled map with the MapTypeId and set it to display.
    map.mapTypes.set('styled_map', styledMapType);
    map.setMapTypeId('styled_map');
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
