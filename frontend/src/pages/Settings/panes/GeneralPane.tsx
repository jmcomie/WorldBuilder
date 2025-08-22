import './PaneStyles.css';
import Checkbox from '../../../components/atoms/Checkbox';
import Input from '../../../components/atoms/Input';
import Select from '../../../components/atoms/Select';
import FormField from '../../../components/molecules/FormField';
import { useLocalStorage } from '../../../shared/hooks';

const GeneralPane = () => {
  const [autoSave, setAutoSave] = useLocalStorage('settings.autoSave', true);
  const [autoSaveInterval, setAutoSaveInterval] = useLocalStorage(
    'settings.autoSaveInterval',
    30
  );
  const [defaultLayout, setDefaultLayout] = useLocalStorage(
    'settings.defaultLayout',
    'force'
  );
  const [animationDuration, setAnimationDuration] = useLocalStorage(
    'settings.animationDuration',
    500
  );

  return (
    <div className="settings-pane">
      <h3 className="settings-pane-title">General Settings</h3>

      <section className="settings-section">
        <h4 className="settings-section-title">Application</h4>

        <FormField
          label="Auto-save episodes"
          description="Automatically save your work as you type"
          orientation="horizontal"
        >
          <Checkbox
            checked={autoSave}
            onChange={(e) => setAutoSave(e.target.checked)}
          />
        </FormField>

        <FormField
          label="Auto-save interval"
          description="How often to save (in seconds)"
          orientation="horizontal"
        >
          <Input
            type="number"
            value={autoSaveInterval}
            onChange={(e) => setAutoSaveInterval(Number(e.target.value))}
            min={5}
            max={300}
          />
        </FormField>
      </section>

      <section className="settings-section">
        <h4 className="settings-section-title">Graph Visualization</h4>

        <FormField
          label="Default layout"
          description="Initial graph layout algorithm"
          orientation="horizontal"
        >
          <Select
            value={defaultLayout}
            onChange={(e) => setDefaultLayout(e.target.value)}
          >
            <option value="force">Force-directed</option>
            <option value="circle">Circle</option>
            <option value="grid">Grid</option>
            <option value="concentric">Concentric</option>
          </Select>
        </FormField>

        <FormField
          label="Animation duration"
          description="Graph animation speed (ms)"
          orientation="horizontal"
        >
          <Input
            type="range"
            value={animationDuration}
            onChange={(e) => setAnimationDuration(Number(e.target.value))}
            min={0}
            max={2000}
            step={100}
          />
        </FormField>
      </section>
    </div>
  );
};

export default GeneralPane;
