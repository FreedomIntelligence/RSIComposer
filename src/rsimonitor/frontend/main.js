/** RSIMonitor single-page app. Chart workspace implementation is adapted from SwanBoard. */
import {createApp} from 'vue';
import {directive} from 'vue-tippy';
import {i18n} from './swanboard/i18n';
import App from './App.vue';
import {start} from './app/store.js';
import './app/app.css';
import './metrics.css';
import 'tippy.js/dist/tippy.css';

const app = createApp(App);
app.use(i18n);                       // SwanBoard chart components
app.directive('tippy', directive);
app.mount('#app');
start();
