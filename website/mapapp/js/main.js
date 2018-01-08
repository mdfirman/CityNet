function initMap() {

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

  var map_inset = new google.maps.Map(document.getElementById('map_inset'), {
    center: london_centre,
    mapTypeControlOptions: {
      mapTypeIds: ['satellite']
    },
    zoom: 18,
    mapTypeId: 'satellite',
    disableDefaultUI: true,
    scrollwheel: false,
    draggable: false
  });

  // We create the map in the json loader callback, beacuse javascript is strange
  loadJSON('assets/maps_style.json', function(response) {
    loaded_json = JSON.parse(response);
    var styledMapType = new google.maps.StyledMapType(loaded_json, {
      name: 'Map'
    });

    var map = new google.maps.Map(document.getElementById('map'), {
      zoom: 11,
      center: london_centre,
      mapTypeControlOptions: {
        mapTypeIds: ['satellite', 'styled_map']
      },
    });

    //Associate the styled map with the MapTypeId and set it to display.
    map.mapTypes.set('styled_map', styledMapType);
    map.setMapTypeId('styled_map');

    var inset_marker = new google.maps.Marker({
      position: this.position,
      map: map_inset,
    });

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
              anchor: new google.maps.Point(sz/2.0, sz/2.0),
            },
            icon_path: icon_url,
            postcode: results.data[i][0],
            wav_path: results.data[i][3],
            im_path: "assets/sites/ims/" + results.data[i][0] + ".jpg",
            chartdata_path: "assets/sites/chartdata/" + results.data[i][0] + ".json",
            description: results.data[i][4],
            sitetype: results.data[i][5],
            isphoto: results.data[i][6],
            startdate: results.data[i][7],
            enddate: results.data[i][8],
            website: results.data[i][9],
            results_data: results.data[i]
          });

          google.maps.event.addListener(marker, 'mouseover', function() {
            this.setIcon({
              url: this.icon_path,
              scaledSize: new google.maps.Size(sz+10, sz+10),
              anchor: new google.maps.Point((sz+10)/2.0, (sz+10)/2.0),
            });
          });

          google.maps.event.addListener(marker, 'mouseout', function() {
            this.setIcon({
              url: this.icon_path,
              scaledSize: new google.maps.Size(sz, sz),
              anchor: new google.maps.Point(sz/2.0, sz/2.0),
            });
          });

          // When marer clicked, the modal is updated then shown
          marker.addListener('click', function() {

            $('#sitetype').html(this.sitetype);
            $('#startdate').html(this.startdate);
            $('#enddate').html(this.enddate);
            $('#sitewebsite').html(this.description);

            if (this.website == ""){
              $('#websitep').hide();
              document.getElementById("sitewebsite").setAttribute("href", "javascript: void(0)");
            }
            else {
              $('#websitep').show();
              document.getElementById("sitewebsite_separate").setAttribute("href", this.website);
              document.getElementById("sitewebsite").setAttribute("href", this.website);
            }

            // if (this.isphoto == 'Y') {
            document.getElementById('site_image').style.display = 'block';
            document.getElementById('site_image').src = this.im_path;
            // }
            // else
            // {
            //   document.getElementById('site_image').style.display = 'none';
            // }

            var chart = Chartkick.charts["minute-data"];
            chart.updateData(this.chartdata_path);
            chart.redraw();

            var myaudio = document.getElementById('audio_source');
            myaudio.src = 'assets/sites/audio_mp3/' + this.wav_path.replace('.wav', '.mp3');

            var audio_container = document.getElementById('audio');
            audio_container.pause()
            audio_container.load()

            $('#myModal').modal('show');
            google.maps.event.trigger(map_inset, "resize");
            inset_marker.setPosition(this.position);
            map_inset.setCenter(this.position);

          });
        }
    	}
    });
  });

  $('#myModal').on('hidden.bs.modal', function () {
    var audio_container = document.getElementById('audio');
    audio_container.pause()
  })


  $("#myModal").on("shown.bs.modal", function () {
    var currentCenter = map_inset.getCenter();
    google.maps.event.trigger(map_inset, "resize");
    map_inset.setCenter(currentCenter);
  });



};

$(document).ready(function(){
  console.log("Loaded")

  google.maps.event.addDomListener(window, "load", initMap);

  $(document).on('click', '#panel-heading', function(e){
    var $this = $(this);
    console.log("Clicked")
  	if(!$this.hasClass('panel-collapsed')) {
  		$this.parents('.panel').find('.panel-body').slideUp();
  		$this.addClass('panel-collapsed');
  		$this.find('i').removeClass('glyphicon-chevron-down').addClass('glyphicon-chevron-up');
      document.getElementById("showhide").innerHTML =  "Show"
  	} else {
  		$this.parents('.panel').find('.panel-body').slideDown();
  		$this.removeClass('panel-collapsed');
  		$this.find('i').removeClass('glyphicon-chevron-up').addClass('glyphicon-chevron-down');
      // $(".showhide").html = "Show";
      document.getElementById("showhide").innerHTML =  "Hide"
  	}
  })

  var is_touch_device = ("ontouchstart" in window) || window.DocumentTouch && document instanceof DocumentTouch;
  $('[data-toggle="popover"]').popover({trigger: is_touch_device ? "click focus" : "hover focus"});
  
  var shiftWindow = function() { scrollBy(0, -60) };
  if (location.hash) shiftWindow();
    window.addEventListener("hashchange", shiftWindow);
  
});


