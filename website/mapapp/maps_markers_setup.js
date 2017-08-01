window.onload = function initMap() {

  function loadJSON(fname, callback) {

    var xobj = new XMLHttpRequest();
    xobj.overrideMimeType("application/json");
    xobj.open('GET', fname, true);
    xobj.onreadystatechange = function() {
      if (xobj.readyState == 4 && xobj.status == "200") {
        callback(xobj.responseText);
      }
    };
    xobj.send(null);
  }

  var london_centre = {lat: 51.489517, lng: -0.124};

  //$("#myModel").load("./assets/sample_site.html");

  // We create the map in the json loader callback, beacuse javascript is strange
  loadJSON('style.json', function(response) {
    loaded_json = JSON.parse(response);
    var styledMapType = new google.maps.StyledMapType(loaded_json, {
      name: 'Map'
    });

    var map = new google.maps.Map(document.getElementById('map'), {
      zoom: 11,
      center: london_centre,
      mapTypeControlOptions: {
        mapTypeIds: ['satellite', 'styled_map']
      }
    });

    //Associate the styled map with the MapTypeId and set it to display.
    map.mapTypes.set('styled_map', styledMapType);
    map.setMapTypeId('styled_map');

    var sz = 35;

    Papa.parse("assets/sites/sites_info.csv", {
    	download: true,
      dynamicTyping: true,
    	complete: function(results) {
        for (i=0; i<results.data.length - 1; i++)
        {
          var icon_url = './assets/sites/charts/' + results.data[i][0] + '.png';

          var marker = new google.maps.Marker({
            position: {lat: results.data[i][1], lng: results.data[i][2]},
            map: map,
            icon: {
              url: icon_url,
              scaledSize: new google.maps.Size(sz, sz),
            },
            icon_path: icon_url,
            postcode: results.data[i][0],
            wav_path: results.data[i][3],
          });

          google.maps.event.addListener(marker, 'mouseover', function() {
            this.setIcon({
              url: this.icon_path,
              scaledSize: new google.maps.Size(sz+10, sz+10),
            });
          });

          google.maps.event.addListener(marker, 'mouseout', function() {
            this.setIcon({
              url: this.icon_path,
              scaledSize: new google.maps.Size(sz, sz),
            });
          });

          // When marer clicked, the modal is updated then shown
          marker.addListener('click', function() {

            $('#main-modal-title').html(this.postcode)

            var chart = Chartkick.charts["minute-data"];
            chart.updateData( [{"data":
              {"0000-00-00T06:00:00.000Z": "0.620196",
              "0000-00-00T20:30:00.000Z": "0.790156",
              "0000-00-00T19:00:00.000Z": "0.888622",
              "0000-00-00T06:30:00.000Z": "0.729119",
              "0000-00-00T12:30:00.000Z": "0.799494",
              "0000-00-00T09:30:00.000Z": "0.767389",
              "0000-00-00T22:30:00.000Z": "0.873739",
              "0000-00-00T15:30:00.000Z": "0.789358",
              "0000-00-00T04:30:00.000Z": "0.236719" } } ] );

            var myaudio = document.getElementById('audio_source');
            myaudio.src = 'assets/sites/audio/' + this.wav_path;

            var audio_container = document.getElementById('audio');
            audio_container.pause()
            audio_container.load()

            $('#myModal').modal('show');
          });
        }
    	}
    });
    //
    // var map_inset = new google.maps.Map(document.getElementById('map_inset'), {
    //   center: london_centre,
    //   zoom: 18,
    //   mapTypeId: 'satellite',
    //   disableDefaultUI: true,
    //   scrollwheel: false,
    //   draggable: false
    // });
    //
    // // Resize map to show on a Bootstrap's modal
    // $('#myModal').on('shown.bs.modal', function() {
    //   var currentCenter = map_inset.getCenter();  // Get current center before resizing
    //   google.maps.event.trigger(map_inset, "resize");
    //   // console.log(london_centre)
    //   map_inset.setCenter(currentCenter); // Re-set previous center
    // });
    //

  });


  $('#myModal').on('hidden.bs.modal', function () {
    var audio_container = document.getElementById('audio');
    audio_container.pause()
  })


}
